# Phase 5.1 Quick Reference Guide

**Phase**: 5.1 - In-Place Operations Consolidation  
**Status**: Foundation Complete (Feb 13) → Next: SSM Implementation (Feb 14+)  
**Target**: 25-30% speedup, 40 KB/step memory reduction

---

## Phase 5.1 Overview

### What Is Phase 5.1?
Systematic replacement of allocating forward passes with in-place variants that write directly to pre-allocated buffers, eliminating ~50% of intermediate allocations.

### Why Phase 5.1?
- **Memory**: 129 KB → ~89 KB per step
- **Speed**: 10-15% per layer, 25-30% total
- **Allocations**: 24 → ~8-10 per step

---

## Implementation Pattern Reference

### Basic Template
```rust
pub fn forward_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
) -> Result<()> {
    let (n, d) = input.dim();
    
    // 1. Validate dimensions
    if output.dim() != (n, d) {
        return Err(Box::new(/* dimension error */));
    }
    
    // 2. Compute without allocating intermediates
    // 3. Write results directly to output buffer
    
    Ok(())
}
```

### With Workspace Reuse
```rust
pub fn forward_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
) -> Result<()> {
    let (n, d) = input.dim();
    if output.dim() != (n, d) {
        return Err(Box::new(/* error */));
    }
    
    // Get workspace buffers
    self.unified_workspace.ensure_capacity(n, 1, d)?;
    let temp_buffer = self.unified_workspace.get_temp_mut();
    
    // Use temp for intermediate, write final to output
    compute_into(input, temp_buffer);
    output.assign(temp_buffer);
    
    Ok(())
}
```

---

## Files to Modify (10 Tasks)

### ✅ Done (Feb 13)
1. `src/domain/layers/components/temporal_processing.rs` - DONE
2. `src/domain/layers/components/common.rs` - DONE

### ⏳ Next (Feb 14-15)
3. `src/domain/layers/ssm/rg_lru.rs` - RgLru + MoHRgLru
4. `src/domain/layers/ssm/mamba.rs` - Mamba + MoHMamba
5. `src/domain/layers/ssm/mamba2.rs` - Mamba2 + MoHMamba2

### ⏳ Later (Feb 18)
6. `src/domain/layers/components/feedforward.rs` - SharedFeedforward
7. `src/domain/layers/richards_glu.rs` - RichardsGlu
8. `src/domain/mixtures/moe.rs` - MixtureOfExperts

### ⏳ Final (Feb 19-20)
9. `src/domain/layers/transformer/block.rs` - TransformerBlock
10. `src/domain/layers/diffusion/block.rs` - DiffusionBlock

---

## Key Methods to Implement

### Per Layer (All Variants)
```
public fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()>
```

### Additionally for Attention
```
pub fn forward_into_with_causal(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
    causal: bool,
)
```

---

## Testing Strategy

### For Each Implementation
1. **Equivalence Test**: Compare forward() vs forward_into()
   ```rust
   let result_alloc = layer.forward(&input);
   let mut result_inplace = Array2::zeros(input.dim());
   layer.forward_into(&input, &mut result_inplace)?;
   
   assert!((result_alloc - &result_inplace).norm() < 1e-5);
   ```

2. **Dimension Validation Test**: Verify error on dimension mismatch
   ```rust
   let mut bad_output = Array2::zeros((999, 999));
   assert!(layer.forward_into(&input, &mut bad_output).is_err());
   ```

3. **Integration Test**: Use in block-level forward pass
   ```rust
   let output = block.forward_into(&input)?;
   ```

---

## Performance Validation

### Benchmarking
```bash
# Run criterion benchmarks
cargo bench --bench forward_into_comparison

# Profile allocations
cargo flamegraph --bin llm_model
```

### Memory Tracking
```bash
# Peak memory usage
/usr/bin/time -v cargo test --lib

# Allocation count (requires custom instrumentation)
```

---

## Debugging Tips

### Common Issues
1. **Dimension Mismatch**: Check output buffer shape before calling forward_into
2. **State Consistency**: Ensure backward() uses same cached values as forward_into()
3. **Numerical Differences**: May be due to different operation order; use < 1e-4 tolerance
4. **Memory Leaks**: Run 1000-step training loop to catch gradual accumulation

### Debug Prints
```rust
eprintln!("Input shape: {:?}", input.dim());
eprintln!("Output shape: {:?}", output.dim());
eprintln!("Workspace capacity: {}", self.unified_workspace.capacity());
```

---

## Build & Test Commands

### Quick Check
```bash
cargo check
```

### Full Build
```bash
cargo build --release
```

### Test All
```bash
cargo test --lib
```

### Test One Component
```bash
cargo test --lib test_rg_lru_forward_into
```

### Specific Test
```bash
cargo test --lib test_rg_lru_forward_into_equivalence -- --exact
```

### Format & Lint
```bash
cargo fmt
cargo clippy --all-targets
```

---

## Phase Timeline

| Date | Focus | Tasks | Tests |
|------|-------|-------|-------|
| Feb 13 | Foundation | Setup, SharedTemporalProcessing | (pending) |
| Feb 14-15 | SSM Variants | RgLru, Mamba, MoH variants | Equivalence |
| Feb 18 | Feedforward | RichardsGlu, MoE | Equivalence |
| Feb 19-20 | Integration | TransformerBlock, DiffusionBlock | Block-level |

---

## Success Criteria Checklist

### Compilation ✅
- [ ] cargo check passes
- [ ] cargo build --release succeeds
- [ ] No new clippy warnings
- [ ] cargo fmt compliant

### Functionality ✅
- [ ] All new forward_into methods implemented
- [ ] All old forward() methods still work
- [ ] Dimension validation works
- [ ] Backward pass still works

### Testing ✅
- [ ] All 476+ existing tests pass
- [ ] All new forward_into tests pass (< 1e-5 error)
- [ ] 100% numerical equivalence with forward()
- [ ] 1000-step stress test without memory leaks

### Performance ✅
- [ ] ≥ 10% latency reduction per layer
- [ ] Memory reduced from 129 KB to ≤ 89 KB/step
- [ ] ≥ 50% fewer allocations per step
- [ ] Cache efficiency validated via profiling

---

## Reference Documentation

| Document | Purpose |
|----------|---------|
| PHASE5_1_IN_PLACE_OPERATIONS_ROADMAP.md | Strategy & rationale |
| PHASE5_1_EXECUTION_PLAN.md | 10-task detailed breakdown |
| CONSOLIDATION_COMPONENTS_MANIFEST.md | Component inventory |
| SESSION_SUMMARY_PHASE5_1_FEB13_2026.md | Session progress |
| This file | Quick reference |

---

## Key Insight: The Forward_Into Pattern

### Traditional (Allocates)
```rust
pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    let hidden = self.compute_hidden(input);      // Alloc 1
    let activated = self.activate(hidden);        // Alloc 2
    let output = self.project(activated);         // Alloc 3
    output
}
```
**Cost**: 3 allocations per call

### In-Place (Zero Allocation)
```rust
pub fn forward_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
) -> Result<()> {
    let hidden = self.unified_workspace.get_hidden_mut();
    self.compute_hidden_into(input, hidden);      // No alloc
    let activated = self.unified_workspace.get_activated_mut();
    self.activate_into(hidden, activated);        // No alloc
    self.project_into(activated, output);         // No alloc
    Ok(())
}
```
**Cost**: 0 new allocations (reuses workspace buffers)

---

## Session Notes (Feb 13, 2026)

**Completed**: Foundation layer (SharedTemporalProcessing + TemporalMixingLayer)  
**Next**: RgLru & Mamba forward_into implementations (Feb 14-15)  
**Status**: On track for 25-30% speedup by Feb 20

---

## Quick Links to Key Code

- **SharedTemporalProcessing**: src/domain/layers/components/temporal_processing.rs#L91-L114
- **TemporalMixingLayer**: src/domain/layers/components/common.rs#L301-L333
- **PolyAttention Reference**: src/domain/attention/poly_attention.rs#L1548-L1641
- **RgLru Target**: src/domain/layers/ssm/rg_lru.rs (forward_into pending)
- **Test Base**: tests/* (will add forward_into_* tests)

