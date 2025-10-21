# Hierarchical Reasoning Model (HRM) - Research Summary

> Archived Notice: HRM has been removed from the codebase as of 2025-10-21.
> This document remains for historical reference only.
> Supported architectures: Transformer and TRM.

## Project: RustGPT - HRM Architecture Integration
### Date: 2025-10-15
### Research Phase: Complete
### Status: Ready for Implementation Planning

---

## Executive Summary

**Hierarchical Reasoning Model (HRM)** is a brain-inspired recurrent architecture published by Sapient Intelligence (June 2025) that achieves exceptional reasoning performance with only 27M parameters and ~1000 training samples. HRM addresses fundamental limitations of standard Transformers by achieving effective computational depth through hierarchical convergence.

### Key Innovation

HRM uses **two interdependent recurrent modules** operating at different timescales:
- **High-Level (H) Module**: Slow, abstract planning (analogous to System 2 thinking)
- **Low-Level (L) Module**: Fast, detailed computations (analogous to System 1 thinking)

### Performance Highlights

| Benchmark | HRM (27M params, 1K samples) | Baseline (Transformer) | o3-mini-high |
|-----------|------------------------------|------------------------|--------------|
| **ARC-AGI-1** | 40.3% | ~15% | 34.5% |
| **Sudoku-Extreme** | ~100% | 0% | 0% |
| **Maze-Hard (30x30)** | ~100% | 0% | 0% |

---

## Architecture Details

### Core Components

```
Input (x)
  ↓
Input Network: fI(x; θI) → x̃
  ↓
┌─────────────────────────────────────┐
│  N High-Level Cycles                │
│  ┌───────────────────────────────┐  │
│  │ T Low-Level Steps per Cycle   │  │
│  │                                │  │
│  │ zL^i = fL(zL^(i-1), zH^(i-1), x̃; θL)  │
│  │                                │  │
│  │ Every T steps:                 │  │
│  │ zH^i = fH(zH^(i-1), zL^(i-1); θH)    │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
  ↓
Output Network: fO(zH^(NT); θO) → ŷ
```

### Mathematical Formulation

**Input Embedding**:
```
x̃ = fI(x; θI)
```

**Low-Level Update** (every timestep i):
```
zL^i = fL(zL^(i-1), zH^(i-1), x̃; θL)
```

**High-Level Update** (every T timesteps):
```
zH^i = {
    fH(zH^(i-1), zL^(i-1); θH)  if i ≡ 0 (mod T)
    zH^(i-1)                     otherwise
}
```

**Output Prediction**:
```
ŷ = fO(zH^(NT); θO)
```

### Hierarchical Convergence

**Key Insight**: Standard RNNs converge too quickly, limiting effective depth.

HRM solves this through **hierarchical convergence**:
1. L-module converges to local equilibrium over T steps
2. H-module updates once, providing new context
3. L-module "resets" and converges to new equilibrium
4. Effective depth: N×T steps (vs T for standard RNN)

### Approximate Gradient (1-Step Gradient)

**Problem**: BPTT requires O(T) memory, biologically implausible

**Solution**: 1-step gradient approximation using Implicit Function Theorem
- Memory: O(1) instead of O(T)
- Gradient path: Output → zH^final → zL^final → Input
- Treats intermediate states as constants (detached)

**PyTorch Implementation Pattern**:
```python
def hrm(z, x, N=2, T=2):
    x = input_embedding(x)
    zH, zL = z
    
    # Forward pass (no grad for N*T-1 steps)
    with torch.no_grad():
        for _i in range(N * T - 1):
            zL = L_net(zL, zH, x)
            if (_i + 1) % T == 0:
                zH = H_net(zH, zL)
    
    # 1-step grad (only last step)
    zL = L_net(zL, zH, x)
    zH = H_net(zH, zL)
    
    return (zH, zL), output_head(zH)
```

### Deep Supervision

**Mechanism**: Multiple forward passes (segments) with supervision at each segment
- Each segment: full N×T timestep forward pass
- Loss computed after each segment
- Hidden state detached between segments (1-step approximation)
- Provides frequent feedback, acts as regularization

### Adaptive Computation Time (ACT)

**Brain-Inspired**: Dynamic "thinking time" based on task complexity
- Q-learning to decide halt/continue
- Q-head predicts Q(halt) and Q(continue) from zH
- Enables "thinking, fast and slow"
- Inference-time scaling: increase Mmax for harder problems

---

## Implementation Architecture for RustGPT

### Module Structure

```rust
pub struct HRMBlock {
    // Core recurrent modules
    low_level: LowLevelModule,    // Fast, detailed computations
    high_level: HighLevelModule,  // Slow, abstract planning
    
    // Hyperparameters
    num_high_cycles: usize,       // N (e.g., 2-4)
    low_steps_per_cycle: usize,   // T (e.g., 2-4)
    
    // Cached states for backward pass
    cached_zL: Option<Array2<f32>>,
    cached_zH: Option<Array2<f32>>,
}

pub struct LowLevelModule {
    // Transformer-based (4-layer encoder)
    transformer: TransformerEncoder,
    embedding_dim: usize,
}

pub struct HighLevelModule {
    // Transformer-based (4-layer encoder)
    transformer: TransformerEncoder,
    embedding_dim: usize,
}
```

### Parameter Count Matching

**Target**: Match Transformer baseline (~27M parameters)

**Current Transformer** (3 blocks):
- Embeddings: vocab_size × 128
- 3 × (Attention + FFN + LayerNorm)
- Output Projection: 128 × vocab_size

**HRM Configuration** (to match):
- Embeddings: vocab_size × 128 (shared)
- Low-Level: 2-layer Transformer (embedding_dim=128, hidden_dim=256)
- High-Level: 2-layer Transformer (embedding_dim=128, hidden_dim=256)
- Output Projection: 128 × vocab_size (shared)
- N=2, T=2 (total 4 effective steps)

### Integration with Existing Codebase

**Zero Impact on Other Architectures**:
1. Add new `HRMBlock` in `src/hrm.rs`
2. Add `LayerEnum::HRMBlock` variant
3. Update `model_builder.rs` with `build_hrm_layers()`
4. Add `ArchitectureType::HRM` to `model_config.rs`
5. All existing Transformer/HyperMixer code unchanged

---

## Key Design Principles

### 1. Hierarchical Processing
- Brain organizes computation across cortical hierarchy
- Higher areas: abstract, slow (theta waves 4-8 Hz)
- Lower areas: detailed, fast (gamma waves 30-100 Hz)

### 2. Temporal Separation
- Different timescales enable stable high-level guidance
- Fast low-level execution under slow high-level control

### 3. Recurrent Connectivity
- Iterative refinement through feedback loops
- Avoids deep credit assignment problem (no BPTT)

### 4. Biological Plausibility
- 1-step gradient: O(1) memory, local learning rules
- Deep supervision: periodic learning (neural oscillations)
- ACT: dynamic resource allocation (System 1/2)

---

## Neuroscience Correspondence

### Dimensionality Hierarchy

**Observation**: Brain regions show PR (Participation Ratio) hierarchy
- Low-level sensory: PR ≈ 30 (low-dimensional)
- High-level associative: PR ≈ 90 (high-dimensional)
- Ratio: ~2.25-3.0

**HRM Replicates This**:
- zL (low-level): PR ≈ 30.22
- zH (high-level): PR ≈ 89.95
- Ratio: ~2.98

**Significance**: Emergent property from training, not architecture
- Untrained HRM: No hierarchy (PR ≈ 40 for both)
- Trained HRM: Clear separation
- Indicates discovery of fundamental organizational principle

---

## Implementation Considerations

### Advantages for RustGPT

1. **Data Efficiency**: Learns from ~1000 samples (vs millions for LLMs)
2. **Parameter Efficiency**: 27M params competitive with billion-param models
3. **Computational Depth**: Effective depth N×T without vanishing gradients
4. **Educational Value**: Brain-inspired, interpretable reasoning process
5. **Rust-Friendly**: No complex dependencies, pure ndarray operations

### Challenges

1. **Training Complexity**: Deep supervision + ACT requires careful implementation
2. **Hyperparameter Tuning**: N, T, Mmax need task-specific tuning
3. **Gradient Approximation**: 1-step gradient is approximation (not exact)
4. **Memory Management**: Need to detach states between segments

### Simplifications for Initial Implementation

**Phase 1** (Minimal HRM):
- Fixed N=2, T=2 (no ACT)
- Single segment (no deep supervision)
- Standard backprop through final step only
- Match Transformer parameter count

**Phase 2** (Full HRM):
- Deep supervision with M segments
- ACT with Q-learning
- Inference-time scaling
- Comprehensive evaluation

---

## References

1. **Primary Paper**: Wang et al., "Hierarchical Reasoning Model", arXiv:2506.21734, June 2025
2. **ARC-AGI Analysis**: https://arcprize.org/blog/hrm-analysis
3. **Sapient Intelligence**: https://sapient.inc
4. **Related Work**:
   - Universal Transformers (Dehghani et al., 2018)
   - Deep Equilibrium Models (Bai et al., 2019)
   - Adaptive Computation Time (Graves, 2016)

---

## Next Steps

1. ✅ Research complete - HRM architecture understood
2. ⏭️ Create detailed implementation plan
3. ⏭️ Implement minimal HRM (Phase 1)
4. ⏭️ Add tests and validation
5. ⏭️ Benchmark against Transformer baseline
6. ⏭️ Implement full HRM with ACT (Phase 2)

---

**Document Status**: COMPLETE  
**Ready for**: Implementation Planning  
**Estimated Implementation Time**: 4-6 hours (Phase 1), 6-8 hours (Phase 2)

