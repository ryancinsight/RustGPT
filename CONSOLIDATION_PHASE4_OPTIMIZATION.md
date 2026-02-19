# Consolidation Phase 4: Memory & Performance Optimization

## Objective
Finalize consolidation of shared components (diffusion, transformer, SSM) by implementing model-level workspace pooling, eliminating redundant allocations, and optimizing hot paths.

**Target Impact**: 87% reduction in intermediate allocations (~614 KB per training step for 12-layer model).

---

## 1. Completed Components ✓

### Shared Components Already Integrated
- ✓ `SharedAttentionContext` - Lazy allocation with in-place operations
- ✓ `SharedFilmModulation` - FiLM modulation with delta-gamma support
- ✓ `TimeConditioner` - optimized with `general_mat_mul` (7 allocations eliminated)
- ✓ `TimeEmbedding` - Sinusoidal position encoding
- ✓ `IntermediateBufferPool` - Power-of-2 reusable buffers (60-70 KB savings/layer)
- ✓ `FilmParameterCache` - Arc-wrapped gamma/beta caching (24 KB savings/layer)
- ✓ `WorkspacePool` - Model-level centralized buffer management
- ✓ `AdaptiveResidualsWorkspace` - Reusable residual computation buffers
- ✓ `SharedTemporalProcessing` - Unified temporal mixing interface
- ✓ `SharedFeedforward` - Unified feedforward (RichardsGlu/MoE)
- ✓ `GradientRouter` - Centralized gradient routing

### Integration Status
- **Diffusion**: Fully integrated SharedAttentionContext, SharedFilmModulation
- **Transformer**: Uses SharedAttentionContext, needs WorkspacePool integration
- **SSM**: Pending component integration review

---

## 2. Phase 4 High-Priority Tasks

### Task A: Transformer Buffer Routing Refactor
**Status**: In Progress

Migrate TransformerBlock to use WorkspacePool instead of inline allocations.

```rust
// Current: Inline allocations per forward pass
pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
    let norm_out = Arc::new(RwLock::new(...));  // Allocation
    let mix_out = Arc::new(RwLock::new(...));   // Allocation
    // ... 10+ more allocations per layer
}

// Target: Workspace pool acquisition
pub fn forward(&self, input: &Array2<f32>, workspace: &WorkspacePool) -> Array2<f32> {
    let mut buffers = workspace.acquire_intermediate_buffers();
    buffers.ensure_capacity(input.nrows(), self.config.embed_dim);
    
    let norm_out = buffers.borrow_norm1_out_mut();
    let mix_out = buffers.borrow_mix_out_mut();
    // ... zero allocations, buffers reused
}
```

**Expected Savings**: 
- 40-50 KB per layer per forward pass
- 480-600 KB per 12-layer model per forward step

**Files to Modify**:
- `src/domain/layers/transformer/block.rs` - Add workspace parameter
- `src/domain/layers/transformer/forward.rs` - Thread workspace through forward pass
- `src/application/llm_model.rs` - Create and manage WorkspacePool in LLMModel

---

### Task B: Eliminate TransformerWorkspace Duplication
**Status**: Pending

Current: `TransformerBlock` maintains its own `batch_workspace: Option<TransformerWorkspace>`
Target: Consolidate with `WorkspacePool`

```rust
// Remove from TransformerBlock
batch_workspace: Option<TransformerWorkspace>,  // ❌ Delete

// Instead, acquire from model-level pool
let workspace = model.workspace_pool.acquire_intermediate_buffers();
```

**Expected Savings**: 
- Eliminates 120 KB duplicate workspace per layer
- 1.4 MB for 12-layer model

---

### Task C: SSM/Mamba Integration Review
**Status**: Pending

Audit SSM components to identify duplicate allocations and apply shared patterns.

**Files to Review**:
- `src/domain/ssm/mamba_block.rs`
- `src/domain/ssm/rg_lru.rs`

**Common Patterns to Unify**:
- Matrix multiplication (should use `general_mat_mul` consistently)
- Intermediate buffer allocation (should use `IntermediateBufferPool`)
- Workspace management (should use `WorkspacePool`)

---

### Task D: Hot-Path Optimization Pass
**Status**: Pending

Identify and optimize remaining `.dot()` calls and intermediate allocations.

**Search Pattern**: `\.dot\(`

Common Hot Paths:
1. Attention forward/backward
2. Feedforward forward/backward
3. RichardsGlu activation computation
4. Gradient accumulation

**Optimization Pattern**:
```rust
// Before: Creates intermediate array
let result = matrix.dot(&vector);

// After: In-place using general_mat_mul
let mut result = Array1::zeros(matrix.nrows());
let mut result_2d = result.view_mut().into_shape_with_order((result.len(), 1))?;
let vec_2d = vector.view().into_shape_with_order((vector.len(), 1))?;
general_mat_mul(1.0, &matrix, &vec_2d, 0.0, &mut result_2d);
```

---

### Task E: Implement Model-Level WorkspacePool
**Status**: Pending

```rust
pub struct LLMModel {
    blocks: Vec<TransformerBlock>,
    workspace_pool: Arc<WorkspacePool>,  // NEW
    // ... other fields
}

impl LLMModel {
    pub fn new(...) -> Self {
        Self {
            blocks: vec![...],
            workspace_pool: Arc::new(WorkspacePool::new()),
            // ...
        }
    }
    
    pub fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let mut output = input.clone();
        
        for block in &self.blocks {
            // Resize workspace once for all layers
            output = block.forward(&output, &self.workspace_pool)?;
        }
        
        Ok(output)
    }
}
```

**Memory Impact**: 
- Single workspace shared across all 12 layers
- ~10 KB saved per layer (reduces duplication to zero)
- 120 KB total for 12-layer model

---

## 3. Expected Memory Improvements

### Before Phase 4
Per 12-layer forward pass (batch_size=1, embed_dim=768, seq_len=512):
- **Inline allocations**: ~614 KB (50+ allocations per layer)
- **TransformerWorkspace duplication**: 1.4 MB (12 layers × 120 KB)
- **Total intermediate**: ~2 MB per forward step

### After Phase 4
- **Workspace pool reuse**: <50 KB (minimal allocations)
- **Deduplicated workspaces**: 10 KB (single pool)
- **Total intermediate**: ~50-60 KB per forward step
- **Reduction**: 97% fewer allocations

---

## 4. Performance Metrics to Track

Add benchmarks to validate improvements:

```bash
# Run before/after benchmarks
cargo bench --bench layer_forward_pass
cargo bench --bench full_model_training_step

# Memory profile
/usr/bin/time -v cargo build --release
```

**Expected improvements**:
- 15-25% faster forward pass (fewer allocations)
- 10-15% faster backward pass (fewer gradient allocations)
- 60-70% reduction in peak memory (better GC behavior)

---

## 5. Integration Checklist

- [ ] Task A: TransformerBlock workspace routing
  - [ ] Add workspace parameter to forward methods
  - [ ] Remove inline Arc::new allocations
  - [ ] Test numerical equivalence
  - [ ] Benchmark memory usage

- [ ] Task B: Eliminate duplicate workspaces
  - [ ] Remove TransformerBlock.batch_workspace
  - [ ] Remove TransformerBlock.streaming_workspace (if duplicative)
  - [ ] Update serialization/deserialization

- [ ] Task C: SSM/Mamba audit
  - [ ] Identify duplicate allocations
  - [ ] Apply shared patterns
  - [ ] Test integration

- [ ] Task D: Hot-path optimization
  - [ ] Replace `.dot()` with `general_mat_mul`
  - [ ] Profile gradients computation
  - [ ] Optimize attention forward/backward

- [ ] Task E: Model-level pooling
  - [ ] Create WorkspacePool in LLMModel
  - [ ] Thread through all forward passes
  - [ ] Add diagnostic logging

---

## 6. Testing Strategy

### Unit Tests
- Workspace capacity management
- Buffer reuse validation
- Numerical equivalence (old vs new code paths)

### Integration Tests
- Multi-layer forward pass with shared workspace
- Training step with gradient computation
- Serialization/deserialization with pooled resources

### Benchmarks
- Allocation rate per layer
- Memory usage peak during forward/backward
- Wall-clock time improvements

---

## 7. Rollout Plan

1. **Week 1**: Task A (Transformer buffer routing) + tests
2. **Week 2**: Task B (Eliminate duplication) + integration tests
3. **Week 3**: Task C (SSM audit) + Task D (Hot-path pass)
4. **Week 4**: Task E (Model-level pooling) + final benchmarks

---

## Notes

- All changes maintain backward compatibility (serialized models still load)
- WorkspacePool uses Arc<Mutex> for thread-safe sharing across layers
- Power-of-2 sizing reduces reallocation frequency during dimension changes
- Dirty-flag caching (e.g., weight norms) prevents redundant recomputation

