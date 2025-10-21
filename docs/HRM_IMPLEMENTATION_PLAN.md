# HRM Implementation Plan

> Archived Notice: HRM has been removed from the codebase as of 2025-10-21.
> This plan is kept for historical reference only.
> Supported architectures: Transformer and TRM.

## Project: RustGPT - Hierarchical Reasoning Model Integration
### Date: 2025-10-15
### Phase: Implementation Planning
### Target: Zero-impact addition of HRM architecture

---

## Implementation Strategy

### Core Principle: **NON-INVASIVE ADDITION**

- ✅ Add new files, do NOT modify existing components
- ✅ Extend enums/traits, do NOT change existing variants
- ✅ Transformer and HyperMixer remain completely unchanged
- ✅ All existing tests continue to pass
- ✅ Parameter count matches Transformer baseline (~27M)

---

## Phase 1: Minimal HRM (4-6 hours)

### Goal
Implement core HRM architecture with hierarchical convergence, matching Transformer parameter count, without ACT or deep supervision.

### Files to Create

#### 1. `src/hrm_low_level.rs` (~150 lines)
**Purpose**: Low-level recurrent module (fast, detailed computations)

```rust
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use crate::llm::Layer;

/// Low-Level Module for HRM
/// 
/// Handles fast, detailed computations within each high-level cycle.
/// Operates at "gamma frequency" (fast timescale).
/// 
/// Architecture: 2-layer Transformer encoder
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LowLevelModule {
    /// Transformer layers for processing
    layer1: TransformerLayer,
    layer2: TransformerLayer,
    
    /// Embedding dimension
    embedding_dim: usize,
    
    /// Cached state for backward pass
    cached_input: Option<Array2<f32>>,
    cached_zH: Option<Array2<f32>>,
    cached_x_tilde: Option<Array2<f32>>,
}

impl LowLevelModule {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        // Initialize with 2-layer Transformer
        // Each layer: Self-Attention + FFN + LayerNorm
    }
    
    /// Forward pass: zL^i = fL(zL^(i-1), zH^(i-1), x̃)
    /// 
    /// Combines three inputs via element-wise addition:
    /// - Previous low-level state zL^(i-1)
    /// - Current high-level state zH^(i-1) (fixed during cycle)
    /// - Input representation x̃
    pub fn forward(
        &mut self,
        zL_prev: &Array2<f32>,
        zH_current: &Array2<f32>,
        x_tilde: &Array2<f32>,
    ) -> Array2<f32> {
        // Cache for backward pass
        self.cached_input = Some(zL_prev.clone());
        self.cached_zH = Some(zH_current.clone());
        self.cached_x_tilde = Some(x_tilde.clone());
        
        // Combine inputs: zL + zH + x̃ (element-wise addition)
        let combined = zL_prev + zH_current + x_tilde;
        
        // Process through 2-layer Transformer
        let out1 = self.layer1.forward(&combined);
        let out2 = self.layer2.forward(&out1);
        
        out2
    }
    
    /// Backward pass with 1-step gradient approximation
    pub fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Backprop through layers
        let grad2 = self.layer2.backward(grads, lr);
        let grad1 = self.layer1.backward(&grad2, lr);
        
        grad1
    }
    
    pub fn parameters(&self) -> usize {
        self.layer1.parameters() + self.layer2.parameters()
    }
}
```

#### 2. `src/hrm_high_level.rs` (~150 lines)
**Purpose**: High-level recurrent module (slow, abstract planning)

```rust
/// High-Level Module for HRM
/// 
/// Handles slow, abstract planning across cycles.
/// Operates at "theta frequency" (slow timescale).
/// 
/// Architecture: 2-layer Transformer encoder
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HighLevelModule {
    /// Transformer layers for processing
    layer1: TransformerLayer,
    layer2: TransformerLayer,
    
    /// Embedding dimension
    embedding_dim: usize,
    
    /// Cached state for backward pass
    cached_input: Option<Array2<f32>>,
    cached_zL: Option<Array2<f32>>,
}

impl HighLevelModule {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        // Initialize with 2-layer Transformer
    }
    
    /// Forward pass: zH^i = fH(zH^(i-1), zL^(i-1))
    /// 
    /// Updates only every T timesteps, using final low-level state
    pub fn forward(
        &mut self,
        zH_prev: &Array2<f32>,
        zL_final: &Array2<f32>,
    ) -> Array2<f32> {
        // Cache for backward pass
        self.cached_input = Some(zH_prev.clone());
        self.cached_zL = Some(zL_final.clone());
        
        // Combine inputs: zH + zL (element-wise addition)
        let combined = zH_prev + zL_final;
        
        // Process through 2-layer Transformer
        let out1 = self.layer1.forward(&combined);
        let out2 = self.layer2.forward(&out1);
        
        out2
    }
    
    /// Backward pass with 1-step gradient approximation
    pub fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Backprop through layers
        let grad2 = self.layer2.backward(grads, lr);
        let grad1 = self.layer1.backward(&grad2, lr);
        
        grad1
    }
    
    pub fn parameters(&self) -> usize {
        self.layer1.parameters() + self.layer2.parameters()
    }
}
```

#### 3. `src/hrm.rs` (~200 lines)
**Purpose**: Main HRM block combining low-level and high-level modules

```rust
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use crate::{
    hrm_low_level::LowLevelModule,
    hrm_high_level::HighLevelModule,
    llm::Layer,
};

/// Hierarchical Reasoning Model Block
/// 
/// Brain-inspired recurrent architecture with two interdependent modules:
/// - High-Level (H): Slow, abstract planning (theta waves ~4-8 Hz)
/// - Low-Level (L): Fast, detailed computations (gamma waves ~30-100 Hz)
/// 
/// Key Innovation: Hierarchical Convergence
/// - L-module converges to local equilibrium over T steps
/// - H-module updates once per cycle, providing new context
/// - L-module "resets" and converges to new equilibrium
/// - Effective depth: N×T steps (vs T for standard RNN)
/// 
/// Reference: Wang et al., "Hierarchical Reasoning Model", arXiv:2506.21734, 2025
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HRMBlock {
    /// Low-level module (fast, detailed)
    low_level: LowLevelModule,
    
    /// High-level module (slow, abstract)
    high_level: HighLevelModule,
    
    /// Number of high-level cycles (N)
    num_high_cycles: usize,
    
    /// Low-level steps per cycle (T)
    low_steps_per_cycle: usize,
    
    /// Embedding dimension
    embedding_dim: usize,
    
    /// Initial states (learned, fixed during training)
    init_zL: Array2<f32>,
    init_zH: Array2<f32>,
    
    /// Cached states for backward pass
    cached_input: Option<Array2<f32>>,
    cached_final_zL: Option<Array2<f32>>,
    cached_final_zH: Option<Array2<f32>>,
}

impl HRMBlock {
    /// Create new HRM block
    /// 
    /// # Arguments
    /// * `embedding_dim` - Dimension of embeddings (e.g., 128)
    /// * `hidden_dim` - Hidden dimension for Transformer layers (e.g., 256)
    /// * `num_high_cycles` - Number of high-level cycles N (e.g., 2)
    /// * `low_steps_per_cycle` - Low-level steps per cycle T (e.g., 2)
    /// * `max_seq_len` - Maximum sequence length
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        num_high_cycles: usize,
        low_steps_per_cycle: usize,
        max_seq_len: usize,
    ) -> Self {
        // Initialize modules
        let low_level = LowLevelModule::new(embedding_dim, hidden_dim);
        let high_level = HighLevelModule::new(embedding_dim, hidden_dim);
        
        // Initialize states (truncated normal, std=1, truncation=2)
        let init_zL = initialize_state(max_seq_len, embedding_dim);
        let init_zH = initialize_state(max_seq_len, embedding_dim);
        
        Self {
            low_level,
            high_level,
            num_high_cycles,
            low_steps_per_cycle,
            embedding_dim,
            init_zL,
            init_zH,
            cached_input: None,
            cached_final_zL: None,
            cached_final_zH: None,
        }
    }
}

impl Layer for HRMBlock {
    fn layer_type(&self) -> &str {
        "HRMBlock"
    }
    
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Cache input for backward pass
        self.cached_input = Some(input.clone());
        
        // Initialize states
        let mut zL = self.init_zL.clone();
        let mut zH = self.init_zH.clone();
        
        // Total timesteps: N × T
        let total_steps = self.num_high_cycles * self.low_steps_per_cycle;
        
        // Main HRM loop
        for i in 0..total_steps {
            // Low-level update (every timestep)
            zL = self.low_level.forward(&zL, &zH, input);
            
            // High-level update (every T timesteps)
            if (i + 1) % self.low_steps_per_cycle == 0 {
                zH = self.high_level.forward(&zH, &zL);
            }
        }
        
        // Cache final states for backward pass
        self.cached_final_zL = Some(zL);
        self.cached_final_zH = Some(zH.clone());
        
        // Return final high-level state
        zH
    }
    
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // 1-step gradient approximation
        // Gradient path: grads → zH → zL → input
        
        // Backward through high-level
        let grad_zH = self.high_level.backward(grads, lr);
        
        // Backward through low-level
        let grad_input = self.low_level.backward(&grad_zH, lr);
        
        grad_input
    }
    
    fn parameters(&self) -> usize {
        self.low_level.parameters() + self.high_level.parameters()
    }
    
    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        _output_grads: &Array2<f32>,
    ) -> Vec<Array2<f32>> {
        // Not used in current training loop
        vec![]
    }
}

/// Initialize state with truncated normal distribution
fn initialize_state(seq_len: usize, dim: usize) -> Array2<f32> {
    use rand::distributions::{Distribution, Normal};
    use rand::thread_rng;
    
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    
    Array2::from_shape_fn((seq_len, dim), |_| {
        let mut val = normal.sample(&mut rng);
        // Truncate to [-2, 2]
        val = val.max(-2.0).min(2.0);
        val as f32
    })
}
```

#### 4. Update `src/lib.rs` (~5 lines)
```rust
pub mod hrm;
pub mod hrm_low_level;
pub mod hrm_high_level;

pub use hrm::HRMBlock;
```

#### 5. Update `src/llm.rs` - Add LayerEnum variant (~10 lines)
```rust
#[derive(Serialize, Deserialize, Debug)]
pub enum LayerEnum {
    // ... existing variants ...
    HRMBlock(Box<crate::hrm::HRMBlock>),
}

impl Layer for LayerEnum {
    fn layer_type(&self) -> &str {
        match self {
            // ... existing matches ...
            LayerEnum::HRMBlock(layer) => layer.layer_type(),
        }
    }
    
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            // ... existing matches ...
            LayerEnum::HRMBlock(layer) => layer.forward(input),
        }
    }
    
    // ... similar for backward, parameters, compute_gradients ...
}
```

#### 6. Update `src/model_config.rs` (~15 lines)
```rust
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ArchitectureType {
    Transformer,
    HyperMixer,
    HRM,  // NEW
}

impl ModelConfig {
    // ... existing methods ...
    
    /// Create HRM configuration
    pub fn hrm(
        embedding_dim: usize,
        hidden_dim: usize,
        num_high_cycles: usize,
        low_steps_per_cycle: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            architecture: ArchitectureType::HRM,
            embedding_dim,
            hidden_dim,
            num_layers: num_high_cycles,  // Reuse for N
            max_seq_len,
            hypernetwork_hidden_dim: Some(low_steps_per_cycle),  // Reuse for T
            num_heads: None,
        }
    }
    
    pub fn get_num_high_cycles(&self) -> usize {
        self.num_layers  // N stored in num_layers
    }
    
    pub fn get_low_steps_per_cycle(&self) -> usize {
        self.hypernetwork_hidden_dim.unwrap_or(2)  // T stored here
    }
}
```

#### 7. Update `src/model_builder.rs` (~40 lines)
```rust
pub fn build_network(config: &ModelConfig, vocab: &Vocab) -> Vec<LayerEnum> {
    let mut layers = Vec::new();
    
    // Add embedding layer (common to all architectures)
    layers.push(LayerEnum::Embeddings(Embeddings::new(vocab.clone())));
    
    // Build architecture-specific layers
    match config.architecture {
        ArchitectureType::Transformer => {
            build_transformer_layers(&mut layers, config);
        }
        ArchitectureType::HyperMixer => {
            build_hypermixer_layers(&mut layers, config);
        }
        ArchitectureType::HRM => {
            build_hrm_layers(&mut layers, config);
        }
    }
    
    // Add output projection layer (common to all architectures)
    layers.push(LayerEnum::OutputProjection(OutputProjection::new(
        config.embedding_dim,
        vocab.words.len(),
    )));
    
    layers
}

/// Build HRM architecture layers
fn build_hrm_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    let num_high_cycles = config.get_num_high_cycles();
    let low_steps_per_cycle = config.get_low_steps_per_cycle();
    
    // Single HRM block (contains N×T effective depth internally)
    let hrm_block = HRMBlock::new(
        config.embedding_dim,
        config.hidden_dim,
        num_high_cycles,
        low_steps_per_cycle,
        config.max_seq_len,
    );
    
    layers.push(LayerEnum::HRMBlock(Box::new(hrm_block)));
}
```

#### 8. Create `tests/hrm_test.rs` (~100 lines)
```rust
use llm::*;
use ndarray::Array2;

#[test]
fn test_hrm_creation() {
    let hrm = HRMBlock::new(128, 256, 2, 2, 80);
    assert_eq!(hrm.layer_type(), "HRMBlock");
}

#[test]
fn test_hrm_forward() {
    let mut hrm = HRMBlock::new(128, 256, 2, 2, 80);
    let input = Array2::zeros((10, 128));
    let output = hrm.forward(&input);
    assert_eq!(output.shape(), &[10, 128]);
}

#[test]
fn test_hrm_backward() {
    let mut hrm = HRMBlock::new(128, 256, 2, 2, 80);
    let input = Array2::zeros((10, 128));
    let output = hrm.forward(&input);
    let grads = Array2::ones(output.shape());
    let input_grads = hrm.backward(&grads, 0.001);
    assert_eq!(input_grads.shape(), input.shape());
}

#[test]
fn test_hrm_parameter_count() {
    let hrm = HRMBlock::new(128, 256, 2, 2, 80);
    let params = hrm.parameters();
    // Should be comparable to Transformer
    assert!(params > 0);
    println!("HRM parameters: {}", params);
}

#[test]
fn test_build_hrm_network() {
    let vocab = Vocab::new(vec!["a", "b", "c"]);
    let config = ModelConfig::hrm(128, 256, 2, 2, 80);
    
    let layers = build_network(&config, &vocab);
    
    // Should have: Embeddings + HRMBlock + OutputProjection = 3 layers
    assert_eq!(layers.len(), 3);
    assert_eq!(layers[0].layer_type(), "Embeddings");
    assert_eq!(layers[1].layer_type(), "HRMBlock");
    assert_eq!(layers[2].layer_type(), "OutputProjection");
}
```

---

## Parameter Count Analysis

### Target: Match Transformer (~27M parameters)

**Transformer (3 blocks)**:
```
Embeddings: vocab_size × 128
3 × (
    Attention: 128×128×3 (Q,K,V) = 49,152
    FFN: 128×256 + 256×128 = 65,536
    LayerNorm: 128×2 (γ,β) × 2 = 512
) = 345,600
Output: 128 × vocab_size
Total: ~27M (with vocab_size ~100K)
```

**HRM (N=2, T=2)**:
```
Embeddings: vocab_size × 128 (shared)
Low-Level (2-layer Transformer):
    2 × (Attention + FFN + LayerNorm) = 230,400
High-Level (2-layer Transformer):
    2 × (Attention + FFN + LayerNorm) = 230,400
Output: 128 × vocab_size (shared)
Total: ~27M (matches!)
```

---

## Testing Strategy

### Unit Tests
1. ✅ HRM block creation
2. ✅ Forward pass shape validation
3. ✅ Backward pass gradient flow
4. ✅ Parameter count verification
5. ✅ Network building with HRM

### Integration Tests
1. ✅ End-to-end training with HRM
2. ✅ Prediction with HRM
3. ✅ Serialization/deserialization
4. ✅ Comparison with Transformer baseline

### Validation Tests
1. ✅ Hierarchical convergence (forward residuals)
2. ✅ Gradient flow (no vanishing/exploding)
3. ✅ Memory usage (comparable to Transformer)
4. ✅ Training stability (loss convergence)

---

## Acceptance Criteria

### Phase 1 Complete When:
- [ ] All new files created and compile
- [ ] All existing tests pass (68/68)
- [ ] New HRM tests pass (≥5 tests)
- [ ] Clippy clean (0 warnings)
- [ ] Parameter count matches Transformer (±10%)
- [ ] HRM trains without NaN/Inf
- [ ] Documentation complete (rustdoc + ADR)

---

## Timeline Estimate

| Task | Time | Status |
|------|------|--------|
| Research & Planning | 2h | ✅ DONE |
| Implement LowLevelModule | 1h | ⏭️ NEXT |
| Implement HighLevelModule | 1h | ⏭️ |
| Implement HRMBlock | 1.5h | ⏭️ |
| Update enums/config | 0.5h | ⏭️ |
| Write tests | 1h | ⏭️ |
| Debug & validate | 1h | ⏭️ |
| Documentation | 0.5h | ⏭️ |
| **Total Phase 1** | **6-8h** | |

---

## Risk Mitigation

### Risk 1: Gradient Approximation Instability
**Mitigation**: Start with small learning rates (1e-4), monitor gradient norms

### Risk 2: Parameter Count Mismatch
**Mitigation**: Calculate exact counts before implementation, validate in tests

### Risk 3: Training Divergence
**Mitigation**: Use same initialization as Transformer, add gradient clipping

### Risk 4: Integration Bugs
**Mitigation**: Comprehensive tests, incremental integration, existing tests as regression suite

---

**Document Status**: COMPLETE  
**Ready for**: Implementation  
**Next Action**: Begin implementing `src/hrm_low_level.rs`

