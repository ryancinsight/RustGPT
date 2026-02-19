# Phase 5.1c: Block Integration - START HERE

**Current Status**: Ready for Implementation  
**Last Updated**: February 13, 2026  
**Progress**: 70% complete (12/14 tasks, Phase 5.1a-b done, 5.1c ready)  

---

## Quick Navigation

**For Quick Start**: 📋 [`PHASE5_1c_QUICK_REFERENCE.md`](PHASE5_1c_QUICK_REFERENCE.md)  
**For Implementation**: 🛠️ [`SESSION_CONTINUATION_PHASE5_1c_STRATEGY.md`](SESSION_CONTINUATION_PHASE5_1c_STRATEGY.md)  
**For Architecture**: 🏗️ [`PHASE5_1c_BLOCK_INTEGRATION_GUIDE.md`](PHASE5_1c_BLOCK_INTEGRATION_GUIDE.md)  
**For Status Check**: ✅ [`PHASE5_1c_SESSION_STATUS.md`](PHASE5_1c_SESSION_STATUS.md)  
**Session Summary**: 📊 [`SESSION_SUMMARY_PHASE5_1c_CONSOLIDATION_FEB13.md`](SESSION_SUMMARY_PHASE5_1c_CONSOLIDATION_FEB13.md)

---

## The Goal (Next 110 minutes)

Convert TransformerBlock batch-path to use in-place workspace buffers instead of allocating temporary arrays.

**Impact**:
- Memory: 20-30 KB/step savings
- Speed: 5-8% per-layer improvement
- Cumulative (Phase 5.1a-c): 70-105 KB/step, 10-15% speedup

---

## What's Done ✅

### All Component Methods Ready
- ✅ Temporal layers: RgLru, Mamba, Mamba2, MoHRgLru, MoHMamba, MoHMamba2, PolyAttention
- ✅ Feedforward layers: RichardsGlu, MixtureOfExperts, SharedFeedforward
- ✅ Test coverage: 504/504 passing

### All Infrastructure in Place
- ✅ UnifiedLayerWorkspace exists with buffers
- ✅ WorkspaceManaged trait defined
- ✅ Build system clean, no errors/warnings

---

## What's Left ⏳ (3 Tasks, 110 minutes)

### Task 1: Workspace Methods (15 min)
**File**: `src/domain/layers/components/unified_layer_workspace.rs`

Add buffer access methods to UnifiedLayerWorkspace:
```rust
pub fn take_norm1_out(&mut self) -> Array2<f32> { ... }
pub fn return_norm1_out(&mut self, buf: Array2<f32>) { ... }
// Repeat for: temporal_out, residual1, norm2_out, ffn_out

pub fn take_all_buffers(&mut self) -> (Array2, Array2, Array2, Array2, Array2) { ... }
pub fn return_all_buffers(&mut self, ...) { ... }
```

### Task 2: RichardsNorm Method (10 min)
**File**: `src/domain/richards/richards_norm.rs`

Add Array2 in-place normalization:
```rust
pub fn normalize_into(&self, input: &Array2<f32>, output: &mut Array2<f32>) {
    let result = self.normalize(input);
    if output.dim() != result.dim() {
        *output = Array2::zeros(result.dim());
    }
    output.assign(&result);
}
```

### Task 3: TransformerBlock Refactor (45 min)
**File**: `src/domain/layers/transformer/block.rs` (lines 763-917)

Replace allocations with workspace buffers:
```rust
fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // ... setup ...
    
    // NEW: Take workspace buffers
    let (mut n1, mut mix, mut r1, mut n2, mut f) = 
        self.unified_workspace.take_all_buffers();
    
    // Replace 6 allocations with forward_into() calls
    self.pre_attention_norm.forward_into(&input_used, &mut n1);
    self.temporal_mixing.forward_into(&n1, &mut mix);
    r1.assign(&mix); r1 += &input_used;
    self.pre_ffn_norm.forward_into(&r1, &mut n2);
    self.feedforward.forward_into(&n2, &mut f);
    f += &r1;
    
    // ... caching ...
    
    // NEW: Return workspace buffers
    self.unified_workspace.return_all_buffers(n1, mix, r1, n2, f.clone());
    
    f
}
```

### Tasks 4-5: Testing & Documentation (40 min)
```bash
cargo test --lib 2>&1        # Verify 504/504 pass
cargo clippy --all-targets   # Check for warnings
cargo bench                  # Optional: benchmark
```

---

## Implementation Approach

### Start Here
```bash
# 1. Open this quick reference
less PHASE5_1c_QUICK_REFERENCE.md

# 2. Read the execution strategy
less SESSION_CONTINUATION_PHASE5_1c_STRATEGY.md

# 3. Locate the files to modify
ls -la src/domain/layers/components/unified_layer_workspace.rs
grep -r "impl RichardsNorm" src/
ls -la src/domain/layers/transformer/block.rs
```

### Then Execute Tasks 1-3
Follow the step-by-step instructions in SESSION_CONTINUATION_PHASE5_1c_STRATEGY.md

### Finally Validate
```bash
cargo build --release 2>&1
cargo test --lib 2>&1
cargo clippy --all-targets 2>&1
```

---

## Success Criteria

✅ When all of these are true:

- [ ] Code builds: `cargo build --release 2>&1` shows no errors
- [ ] Tests pass: `cargo test --lib 2>&1` shows 504/504 passing
- [ ] No warnings: `cargo clippy --all-targets 2>&1` shows clean output
- [ ] Workspace methods added to UnifiedLayerWorkspace
- [ ] RichardsNorm has normalize_into() method
- [ ] TransformerBlock::forward() uses workspace buffers
- [ ] Performance benchmark shows 5-8% improvement or no regression
- [ ] CONSOLIDATION_COMPONENTS_MANIFEST.md marked Phase 5.1c complete

---

## File Structure

```
Phase 5.1c Documentation:
├── PHASE5_1c_START_HERE.md                    ← You are here
├── PHASE5_1c_QUICK_REFERENCE.md               ← 3-step summary
├── SESSION_CONTINUATION_PHASE5_1c_STRATEGY.md ← Detailed execution
├── PHASE5_1c_BLOCK_INTEGRATION_GUIDE.md       ← Architecture guide
├── PHASE5_1c_SESSION_STATUS.md                ← Readiness checklist
└── SESSION_SUMMARY_PHASE5_1c_CONSOLIDATION_FEB13.md

Code Files to Modify:
├── src/domain/layers/components/unified_layer_workspace.rs
├── src/domain/richards/richards_norm.rs
└── src/domain/layers/transformer/block.rs

Reference Documentation:
└── CONSOLIDATION_COMPONENTS_MANIFEST.md       ← Overall progress
```

---

## Time Breakdown

```
Task 1: Workspace methods          15 min  ⏳
Task 2: RichardsNorm method        10 min  ⏳
Task 3: TransformerBlock refactor  45 min  ⏳
Task 4: Testing                    20 min  ⏳
Task 5: Documentation               5 min  ⏳
Misc (reading, fixing)             15 min  ⏳
═══════════════════════════════════════════
Total                             110 min  (1h 50 min)
```

---

## Key Patterns to Remember

### Pattern 1: Workspace Buffer Lifecycle
```rust
// Take
let mut buf = workspace.take_norm1_out();
// Use
self.layer.forward_into(&input, &mut buf);
// Return
workspace.return_norm1_out(buf);
```

### Pattern 2: Forward Into
```rust
// Instead of:
// let output = layer.forward(&input);

// Do:
layer.forward_into(&input, &mut output);
```

### Pattern 3: Caching for Backward
```rust
// Keep Arc clones for backward pass
*self.cached_intermediates.write().unwrap() = Some(CachedIntermediates {
    norm1_out: Arc::new(norm1_out.clone()),  // Clone for cache
    // ...
});
```

---

## Troubleshooting Quick Links

| Problem | Solution | Doc |
|---------|----------|-----|
| "Cannot move out of self" | Use take()/return() pattern | SESSION_CONTINUATION_PHASE5_1c_STRATEGY.md#Troubleshooting |
| Type mismatch Array2 vs Option | Ensure take_* returns Array2 | PHASE5_1c_QUICK_REFERENCE.md#Common-Pitfalls |
| Compilation errors | Check borrowing rules | PHASE5_1c_BLOCK_INTEGRATION_GUIDE.md#Code-Pattern-Reference |
| Tests failing | Verify cached intermediates | SESSION_CONTINUATION_PHASE5_1c_STRATEGY.md#Task-4 |
| No performance gain | Check buffer reuse | PHASE5_1c_QUICK_REFERENCE.md#Common-Pitfalls |

---

## Next Phase (Preview)

Once Phase 5.1c completes, Phase 5.2 continues:

### Phase 5.2a: Global Buffer Pooling
- Consolidate all layer workspace pools
- Expected: 30% fragmentation reduction

### Phase 5.2b: Selective Gradients
- Skip gradient computation for frozen layers
- Expected: 20-30 KB/step additional savings

### Phase 5.2c: Mixed Precision
- f16 for historical matrices
- Expected: 50% reduction for large contexts

---

## Questions?

📍 **Thread Reference**: T-019c56ff-fa0e-70a8-895a-9a4f2330e303  
📊 **Manifest**: CONSOLIDATION_COMPONENTS_MANIFEST.md  
🔗 **Build Guide**: AGENTS.md  

---

**Status**: ✅ Ready to implement. Choose a reference document above and begin!
