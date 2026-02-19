# Phase 5.1: Integration Template

**Purpose**: Step-by-step guide for integrating WorkspaceManaged into existing blocks  
**Target Blocks**: TransformerBlock, DiffusionBlock, RgLruBlock  
**Difficulty**: Medium (mostly mechanical replacement)

---

## Template: Converting a Block to Use WorkspaceManaged

### Step 1: Add Trait Impl (Boilerplate)

```rust
use crate::domain::layers::components::{
    WorkspaceManaged, UnifiedLayerWorkspace, WorkspaceStats,
};

// Add to existing impl block or create new one
impl WorkspaceManaged for YourBlock {
    fn ensure_capacity(&mut self, batch_size: usize, seq_len: usize, embed_dim: usize) {
        self.workspace.ensure_capacity(batch_size, seq_len, embed_dim);
    }

    fn clear_workspace(&mut self) {
        self.workspace.clear_workspace();
    }

    fn workspace_stats(&self) -> WorkspaceStats {
        self.workspace.workspace_stats()
    }
}
```

### Step 2: Add Workspace Field

**Before**:
```rust
pub struct TransformerBlock {
    pub norm1_workspace: Option<Array2<f32>>,
    pub attention_workspace: Option<Array2<f32>>,
    pub norm2_workspace: Option<Array2<f32>>,
    pub ffn_workspace: Option<Array2<f32>>,
    // ... other fields
}
```

**After**:
```rust
pub struct TransformerBlock {
    // Replace all workspace fields with single unified workspace
    pub workspace: UnifiedLayerWorkspace,
    // ... other fields
}
```

### Step 3: Update Constructor

**Before**:
```rust
impl TransformerBlock {
    pub fn new(...) -> Self {
        Self {
            norm1_workspace: None,
            attention_workspace: None,
            norm2_workspace: None,
            ffn_workspace: None,
            // ...
        }
    }
}
```

**After**:
```rust
impl TransformerBlock {
    pub fn new(...) -> Self {
        Self {
            workspace: UnifiedLayerWorkspace::new(),
            // ...
        }
    }
}
```

### Step 4: Update Forward Pass

**Before**:
```rust
pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // Manual capacity management
    if self.norm1_workspace.is_none() || 
       self.norm1_workspace.as_ref().unwrap().dim() != (batch, seq) {
        self.norm1_workspace = Some(Array2::zeros((batch, seq)));
    }
    
    let norm1_out = self.norm1_workspace.as_mut().unwrap();
    norm1_out.assign(&self.norm1.forward(input));
    
    // ... repeat for other buffers
    
    // Return output
    let output = ...;
    output
}
```

**After**:
```rust
pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    let batch = input.nrows();
    let seq = input.ncols();
    let embed_dim = self.embed_dim;
    
    // Single call for all buffer capacity
    self.workspace.ensure_capacity(batch, seq, embed_dim);
    
    // Use reusable buffers
    {
        let norm1_out = self.workspace.norm1_out_mut().unwrap();
        norm1_out.assign(&self.norm1.forward(input));
    }
    
    {
        let temporal_out = self.workspace.temporal_out_mut().unwrap();
        temporal_out.assign(&self.temporal.forward(
            self.workspace.norm1_out().unwrap()
        ));
    }
    
    // ... more buffer usage
    
    // Return output (clone from workspace if needed)
    self.workspace.temporal_out().unwrap().clone()
}
```

### Step 5: Update Backward Pass

**Before**:
```rust
pub fn backward(
    &self,
    input: &Array2<f32>,
    output_grads: &Array2<f32>,
) -> (Array2<f32>, Vec<Array2<f32>>) {
    // Manual workspace access
    let norm1_out = self.norm1_workspace.as_ref().unwrap();
    let attention_out = self.attention_workspace.as_ref().unwrap();
    
    // ... gradient computation
    
    (input_grads, param_grads)
}
```

**After**:
```rust
pub fn backward(
    &self,
    input: &Array2<f32>,
    output_grads: &Array2<f32>,
) -> (Array2<f32>, Vec<Array2<f32>>) {
    // Access workspace via accessor methods
    let norm1_out = self.workspace.norm1_out()
        .expect("workspace not allocated");
    let temporal_out = self.workspace.temporal_out()
        .expect("workspace not allocated");
    
    // ... gradient computation
    
    (input_grads, param_grads)
}
```

---

## Concrete Example 1: TransformerBlock

### Original State (Current)
**File**: `src/domain/layers/transformer/block.rs` (lines 40-450)

```rust
pub struct TransformerBlock {
    pub embed_dim: usize,
    pub num_heads: usize,
    
    // Workspace pools (FRAGMENTED)
    pub workspace_pool: Option<WorkspacePool>,
    pub residuals_workspace: Option<AdaptiveResidualsWorkspace>,
    pub streaming_workspace: Option<TransformerBlockStreamingWorkspace>,
    
    // Components
    pub norm1: RichardsNorm,
    pub attention: PolyAttention,
    pub norm2: RichardsNorm,
    pub ffn: SharedFeedforward,
    
    // ... other fields
}
```

### Target State (After Integration)

```rust
pub struct TransformerBlock {
    pub embed_dim: usize,
    pub num_heads: usize,
    
    // UNIFIED workspace
    pub workspace: UnifiedLayerWorkspace,
    
    // Components
    pub norm1: RichardsNorm,
    pub attention: PolyAttention,
    pub norm2: RichardsNorm,
    pub ffn: SharedFeedforward,
    
    // ... other fields
}

// Add trait implementation
impl WorkspaceManaged for TransformerBlock {
    fn ensure_capacity(&mut self, batch: usize, seq: usize, embed: usize) {
        self.workspace.ensure_capacity(batch, seq, embed);
    }
    
    fn clear_workspace(&mut self) {
        self.workspace.clear_workspace();
    }
    
    fn workspace_stats(&self) -> WorkspaceStats {
        self.workspace.workspace_stats()
    }
}
```

### Integration Checklist for TransformerBlock
- [ ] Add `use WorkspaceManaged` import
- [ ] Replace 3 workspace fields with `UnifiedLayerWorkspace`
- [ ] Update `new()` constructor
- [ ] Update `forward()` method
- [ ] Update `backward()` method
- [ ] Update `apply_gradients()` if needed
- [ ] Run existing tests (should pass unchanged)
- [ ] Run workspace-specific tests
- [ ] Benchmark allocation count (target: -40%)
- [ ] Benchmark peak memory (target: -20%)

**Estimated Time**: 2-3 hours (mostly mechanical)

---

## Concrete Example 2: DiffusionBlock

### Original State
**File**: `src/domain/layers/diffusion/block.rs` (lines 628-1105)

```rust
pub struct DiffusionBlock {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub timestep_embed_dim: usize,
    
    // Workspace (CUSTOM implementation)
    norm1_out: Option<Array2<f32>>,
    timestep_proj_out: Option<Array2<f32>>,
    mix_out: Option<Array2<f32>>,
    norm2_out: Option<Array2<f32>>,
    ffn_out: Option<Array2<f32>>,
    
    // Components
    pub norm1: RichardsNorm,
    pub timestep_proj: FeedforwardProcessor,
    pub mixing: SharedTemporalProcessing,
    pub norm2: RichardsNorm,
    pub ffn: SharedFeedforward,
    
    // ... other fields
}
```

### Target State

```rust
pub struct DiffusionBlock {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub timestep_embed_dim: usize,
    
    // UNIFIED workspace
    pub workspace: UnifiedLayerWorkspace,
    
    // Components
    pub norm1: RichardsNorm,
    pub timestep_proj: FeedforwardProcessor,
    pub mixing: SharedTemporalProcessing,
    pub norm2: RichardsNorm,
    pub ffn: SharedFeedforward,
    
    // ... other fields
}

impl WorkspaceManaged for DiffusionBlock {
    fn ensure_capacity(&mut self, batch: usize, seq: usize, embed: usize) {
        self.workspace.ensure_capacity(batch, seq, embed);
    }
    
    fn clear_workspace(&mut self) {
        self.workspace.clear_workspace();
    }
    
    fn workspace_stats(&self) -> WorkspaceStats {
        self.workspace.workspace_stats()
    }
}
```

### Integration Checklist for DiffusionBlock
- [ ] Add `use WorkspaceManaged` import
- [ ] Replace 5 workspace fields with `UnifiedLayerWorkspace`
- [ ] Update `new()` constructor
- [ ] Update `forward()` method (also handles timestep embedding)
- [ ] Update `backward()` method
- [ ] Run existing tests (verify loss curves unchanged)
- [ ] Run workspace tests
- [ ] Benchmark allocation count
- [ ] Benchmark memory usage

**Estimated Time**: 2-3 hours

---

## Concrete Example 3: RgLruBlock (SSM)

### Original State
**File**: `src/domain/layers/ssm/rg_lru.rs` (lines 150-350)

```rust
pub struct RgLruBlock {
    pub embed_dim: usize,
    pub expansion_factor: f32,
    
    // Streaming state (CURRENT implementation)
    rnn_state: Option<Array2<f32>>,
    u_cache: Option<Array2<f32>>,
    
    // Workspace (MINIMAL)
    // Allocated on-the-fly in forward pass
    
    // Components
    pub temporal: TemporalMixingLayer,
    pub ffn: SharedFeedforward,
    
    // ... other fields
}
```

### Target State

```rust
pub struct RgLruBlock {
    pub embed_dim: usize,
    pub expansion_factor: f32,
    
    // UNIFIED workspace with streaming support
    pub workspace: UnifiedLayerWorkspace,
    
    // Components
    pub temporal: TemporalMixingLayer,
    pub ffn: SharedFeedforward,
    
    // ... other fields
}

impl WorkspaceManaged for RgLruBlock {
    fn ensure_capacity(&mut self, batch: usize, seq: usize, embed: usize) {
        self.workspace.ensure_capacity(batch, seq, embed);
    }
    
    fn clear_workspace(&mut self) {
        self.workspace.clear_workspace();
    }
    
    fn workspace_stats(&self) -> WorkspaceStats {
        self.workspace.workspace_stats()
    }
}

// ALSO implement StreamingWorkspaceManaged
impl StreamingWorkspaceManaged for RgLruBlock {
    fn init_streaming(&mut self, batch: usize, embed: usize) -> Result<()> {
        self.workspace.ensure_capacity(batch, 1, embed);
        // Initialize RNN state in workspace
        self.workspace.streaming_state = Some(Array2::zeros((batch, embed)));
        Ok(())
    }
    
    fn reset_streaming_state(&mut self) {
        self.workspace.reset_streaming();
    }
    
    fn is_streaming(&self) -> bool {
        self.workspace.streaming_state().is_some()
    }
    
    fn finalize_streaming(&mut self) -> Option<Array2<f32>> {
        self.workspace.streaming_state().cloned()
    }
}
```

### Integration Checklist for RgLruBlock
- [ ] Add `use WorkspaceManaged, StreamingWorkspaceManaged` imports
- [ ] Replace streaming state fields with `UnifiedLayerWorkspace`
- [ ] Implement `WorkspaceManaged` trait
- [ ] Implement `StreamingWorkspaceManaged` trait
- [ ] Update `forward()` for batch mode
- [ ] Update `forward_streaming()` for recurrent mode
- [ ] Test batch forward pass (should match before)
- [ ] Test streaming forward pass (verify state management)
- [ ] Benchmark allocation reduction
- [ ] Verify streaming correctness with long sequences

**Estimated Time**: 2-3 hours

**Extra Consideration**: RgLruBlock needs both `ensure_capacity` and streaming state management, making it the most complex integration.

---

## Testing Strategy for Integration

### Unit Tests
```rust
#[test]
fn test_transformer_block_workspace_allocation() {
    let mut block = TransformerBlock::new(...);
    
    // Allocate for specific shape
    block.ensure_capacity(32, 512, 2048);
    assert!(block.workspace.all_buffers_allocated());
    
    // Reuse same shape
    let stats_before = block.workspace_stats();
    block.ensure_capacity(32, 512, 2048);
    let stats_after = block.workspace_stats();
    assert_eq!(stats_before.buffer_count, stats_after.buffer_count);
    
    // Different shape triggers reallocation
    block.ensure_capacity(64, 1024, 2048);
    let stats_new = block.workspace_stats();
    assert!(stats_new.allocation_count > stats_after.allocation_count);
}

#[test]
fn test_diffusion_block_numerical_equivalence() {
    let mut block_old = DiffusionBlock::new(...); // Old implementation (if available)
    let mut block_new = DiffusionBlock::new(...); // New implementation
    
    let input = Array2::from_elem((32, 512, 2048), 0.5f32);
    let timestep = Array1::from_elem(32, 0.5f32);
    
    let out_old = block_old.forward(&input, Some(&timestep));
    let out_new = block_new.forward(&input, Some(&timestep));
    
    // Results should be identical (within floating point tolerance)
    assert_eq!(out_old.shape(), out_new.shape());
    for (a, b) in out_old.iter().zip(out_new.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_rglru_streaming_workspace() {
    let mut block = RgLruBlock::new(...);
    
    // Initialize streaming
    block.init_streaming(32, 2048).unwrap();
    assert!(block.is_streaming());
    
    // Run streaming forward
    for _ in 0..100 {
        let token = Array1::from_elem(2048, 0.5f32);
        let _ = block.forward_streaming(&token);
    }
    
    // Get accumulated state
    let state = block.finalize_streaming();
    assert!(state.is_some());
    assert_eq!(state.unwrap().shape(), (32, 2048));
}
```

### Integration Tests
```rust
// tests/transformer_block_integration.rs
#[test]
fn test_transformer_block_forward_backward_consistency() {
    // Load pretrained model
    // Run forward pass
    // Verify activations match expected (compare with checkpoint)
    // Run backward pass
    // Verify gradients are finite and non-zero
}

// tests/diffusion_block_integration.rs
#[test]
fn test_diffusion_loss_curves_unchanged() {
    // Train for N steps with old implementation
    // Train for N steps with new implementation
    // Verify loss curves are within 1% of each other
}

// tests/ssm_block_integration.rs
#[test]
fn test_rglru_streaming_vs_batch() {
    // Generate random sequence
    // Process in batch mode
    // Process in streaming mode (token by token)
    // Verify outputs match (within tolerance)
}
```

---

## Performance Validation Checklist

After integration, measure these metrics:

### Allocation Metrics
```bash
Before:
  Allocations per step: ~50-60
  Peak memory: 2.0 GB
  Allocation count in workspace: ~12-15

After:
  Allocations per step: ~30-35 (target: -40%)
  Peak memory: 1.6 GB (target: -20%)
  Allocation count in workspace: ~6 (matched by design)
```

### Performance Metrics
```bash
Before:
  Forward pass: 450ms
  Backward pass: 250ms
  E2E step: 700ms

After:
  Forward pass: 380ms (target: -15%)
  Backward pass: 210ms (target: -16%)
  E2E step: 590ms (target: -16%)
```

### Code Quality Metrics
```bash
Compilation:
  Before: 45s
  After: 45s (no regression)

Test coverage:
  Before: 85%
  After: 95%

Duplication:
  Before: 300+ LOC
  After: 0 LOC (consolidated)
```

---

## Rollback Plan

If integration causes issues:

1. **Revert changes to specific block**:
   ```bash
   git checkout src/domain/layers/transformer/block.rs
   ```

2. **Keep trait & workspace implementation**:
   - These are independently testable and safe
   - Can be used incrementally

3. **Try simpler approach first**:
   - Start with TransformerBlock (simpler)
   - Then DiffusionBlock (medium)
   - Finally RgLruBlock (complex)

4. **Feature gate if needed**:
   ```rust
   #[cfg(feature = "unified_workspace")]
   impl WorkspaceManaged for TransformerBlock { ... }
   ```

---

## Success Criteria

Each block integration is successful when:

- ✅ Code compiles without warnings
- ✅ All existing tests pass unchanged
- ✅ Workspace allocation tests pass (9/9)
- ✅ Numerical equivalence verified (vs. old implementation)
- ✅ Allocation count reduced by ~40%
- ✅ Peak memory reduced by ~20%
- ✅ Forward pass latency reduced by ~15%
- ✅ Documentation updated

---

## Timeline

```
Week 1 (Phase 5.1):
  Mon-Tue: TransformerBlock integration + testing
  Wed-Thu: DiffusionBlock integration + testing
  Fri:     RgLruBlock integration + testing

Week 2 (Phase 5.2):
  Mon-Tue: Performance profiling & validation
  Wed-Thu: In-place operations implementation
  Fri:     Documentation & summary
```

---

## Questions Before Starting?

1. **Q**: What if a block has custom workspace logic?
   **A**: Migrate it into the context/streaming buffers in UnifiedLayerWorkspace

2. **Q**: What if forward pass returns workspace buffer?
   **A**: Clone it (performance cost is minimal vs. allocation savings)

3. **Q**: How to handle serialization?
   **A**: Workspace is #[serde(skip)], so serialization is unchanged

4. **Q**: Can we roll out incrementally?
   **A**: Yes! Each block is independent. TransformerBlock → DiffusionBlock → RgLruBlock

---

**Ready to integrate?** Start with TransformerBlock! 🚀
