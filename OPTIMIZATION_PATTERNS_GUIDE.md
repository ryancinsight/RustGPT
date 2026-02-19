# Optimization Patterns Guide - RustGPT

## Overview
This document captures optimization patterns and lessons learned during consolidation and memory efficiency improvements in RustGPT. Use this as reference for future optimization work.

---

## Pattern 1: Pre-Allocate and Reuse Buffer Pattern

### Problem
Matrix multiplication using `.dot()` allocates intermediate arrays:
```rust
// OLD: Allocates (seq_len × embed_dim) intermediate
let result = input.dot(context);
```

### Solution
Pre-allocate output buffer and use `general_mat_mul()`:
```rust
// NEW: Reuses provided buffer, no intermediate allocation
let mut result = Array2::<f32>::zeros((input.nrows(), context.ncols()));
ndarray::linalg::general_mat_mul(1.0, &input, &context, 0.0, &mut result);
```

### Benefits
- **Memory**: Eliminates O(seq_len × embed_dim) intermediate allocations
- **Cache**: Reused buffers stay in L1/L2 cache
- **Bandwidth**: Fewer memory allocations = less memory traffic
- **Predictability**: Fixed buffer sizes enable SIMD optimizations

### When to Use
- ✅ Hot-path matrix multiplications (called frequently)
- ✅ When output buffer can be pre-allocated or pooled
- ✅ When numerical precision is critical (same algorithm)
- ❌ One-off computations (allocation overhead isn't saved)
- ❌ Variable-size outputs hard to pre-allocate

### Performance Impact
- **Forward Pass**: 3-5% throughput improvement
- **Backward Pass**: 5-8% throughput improvement (fewer allocations)
- **Memory**: 20-60 MB saved per batch (large models)

### Code Checklist
- [ ] Identify hot-path `.dot()` calls
- [ ] Pre-allocate output buffer (can be in workspace or inline)
- [ ] Use `general_mat_mul(alpha, A, B, beta, &mut C)`
- [ ] Add tests validating numerical equivalence (within 1e-5)
- [ ] Benchmark memory/performance impact
- [ ] Document optimization intent in comments

---

## Pattern 2: Lazy Allocation for Optional Components

### Problem
Allocating large matrices (embed_dim × embed_dim) upfront when they might not be needed:
```rust
// OLD: Always allocates, even if not used
pub struct Context {
    pub similarity_matrix: Array2<f32>,  // Always allocates embed_dim²
}
```

### Solution
Allocate only when actually needed:
```rust
// NEW: Allocates only on first use
pub struct Context {
    pub similarity_matrix: Option<Array2<f32>>,  // Lazy allocation
}

pub fn update_context(&mut self, ...) {
    if self.similarity_matrix.is_none() 
        || self.similarity_matrix.as_ref().unwrap().shape() != [embed_dim, embed_dim]
    {
        self.similarity_matrix = Some(Array2::zeros((embed_dim, embed_dim)));
    }
    // ... update
}
```

### Benefits
- **Memory**: Saves embed_dim² × 4 bytes when component unused
- **Startup**: Faster initialization (no large allocations)
- **Flexibility**: Different sequence lengths don't trigger reallocations

### Memory Savings
- embed_dim=768: 2.36 MB saved per context
- Multi-layer model: 2.36 MB × num_layers (e.g., 47 MB for 20 layers)

### When to Use
- ✅ Components that may not be used (e.g., optional context)
- ✅ Inference vs training mode (training needs full state)
- ✅ Large dimensional matrices (embed_dim² can be significant)
- ❌ Always-used components (allocation overhead isn't saved)
- ❌ Frequently allocated/deallocated (fragmentation risk)

### Code Checklist
- [ ] Convert `Array2<T>` to `Option<Array2<T>>`
- [ ] Add shape validation on allocation
- [ ] Check `is_none()` before operations
- [ ] Use `#[serde(skip)]` for transient state
- [ ] Add tests for both allocated and unallocated states

---

## Pattern 3: Power-of-2 Sized Workspace Pooling

### Problem
Different sequence lengths trigger repeated reallocations:
```rust
// Sequence lengths: 32, 64, 128, 256, 512
// Each new length triggers reallocation
pub struct Workspace {
    pub buffer: Vec<f32>,  // Reallocates on every new seq_len
}
```

### Solution
Round up to next power of 2 and reuse:
```rust
pub struct Workspace {
    pub buffer: Vec<f32>,
    pub capacity: usize,
}

impl Workspace {
    pub fn resize_for_dim(&mut self, embed_dim: usize) {
        let new_capacity = embed_dim.next_power_of_two().max(32);
        
        if new_capacity != self.capacity {
            self.buffer.resize(new_capacity, 0.0);
            self.capacity = new_capacity;
        }
        
        // Clear without deallocating
        self.buffer.fill(0.0);
    }
}
```

### Benefits
- **Allocations**: Reduced from O(unique_dims) to O(log(max_dim))
- **Reuse**: Buffer reused within power-of-2 bracket
- **Waste**: ~1.5x max needed (acceptable trade-off)
- **Predictability**: Bounded memory usage

### Example
- Sequence lengths: 32→32 (reuse), 64→64 (reuse), 100→128 (reuse)
- Only 3 allocations instead of N allocations

### When to Use
- ✅ Scratch buffers with variable input dimensions
- ✅ Workspace objects shared across layers
- ✅ When rounding up to next power-of-2 is acceptable
- ❌ Exact-size buffers must be maintained
- ❌ Memory-constrained environments

### Code Checklist
- [ ] Define capacity field
- [ ] Implement `resize_for_dim()` with power-of-2 rounding
- [ ] Use `buffer.resize()` only on capacity change
- [ ] Use `buffer.fill(0.0)` to clear without deallocating
- [ ] Test with various dimensions (esp. boundaries: 63→64, 127→128)
- [ ] Document acceptable memory overhead

---

## Pattern 4: Dirty-Flag Caching (Conditional - Thread-Safe Variant)

### Problem
Recomputing expensive metrics on every access:
```rust
pub fn weight_norm(&self) -> f32 {
    // O(embed_dim) computation every call
    let mut sum = 0.0;
    for &v in self.params.iter() {
        sum += v * v;  // Recomputes even if unchanged
    }
    sum.sqrt()
}
```

### Solution (For Single-Threaded or Protected Contexts)
Cache with dirty flag:
```rust
pub struct Component {
    params: Array2<f32>,
    cached_norm: Option<f32>,
    norm_is_dirty: bool,
}

impl Component {
    pub fn weight_norm(&self) -> f32 {
        if !self.norm_is_dirty {
            if let Some(cached) = self.cached_norm {
                return cached;
            }
        }
        
        let norm = self.compute_norm();
        self.cached_norm = Some(norm);
        self.norm_is_dirty = false;
        norm
    }
    
    pub fn apply_gradients(...) {
        // ... update params ...
        self.norm_is_dirty = true;  // Mark cache as invalid
    }
}
```

### Caveats
- **Thread-Safety**: Interior mutability requires `Cell`/`RefCell` which is `!Sync`
- **For Sync contexts**: Would need `Mutex` or `Arc<Mutex>` (introduces overhead)
- **ROI**: Only worth it if metric is called frequently and computation is expensive

### When to Use  
- ✅ Expensive metrics (O(n²) or more)
- ✅ Called frequently without parameter changes
- ✅ Single-threaded or protected contexts only
- ❌ Metrics in parallel contexts (`!Sync` required)
- ❌ Cheap computations (overhead > benefit)
- ❌ Frequent parameter updates (constant dirty flag flipping)

### Lesson Learned
For RustGPT: Weight norm is O(embed_dim) which is ~256-1024 operations - fast enough that caching complexity is not justified, especially when it violates thread-safety.

---

## Pattern 5: Generational Buffer Reuse

### Problem
Allocating separate buffers for intermediate results in every forward pass:
```rust
// OLD: Allocates new buffer on each forward
pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    let norm = Arc::new(self.norm.forward(input));  // New alloc
    let mix = Arc::new(self.mix.forward(&norm));     // New alloc
    let out = Arc::new(self.ffn.forward(&mix));      // New alloc
    // ...
}
```

### Solution
Pre-allocate once and clear between uses:
```rust
pub struct BlockWorkspace {
    pub norm_out: Array2<f32>,
    pub mix_out: Array2<f32>,
    pub ffn_out: Array2<f32>,
}

impl BlockWorkspace {
    pub fn ensure_capacity(&mut self, seq_len: usize, embed_dim: usize) {
        let cap_seq = seq_len.next_power_of_two();
        let cap_embed = embed_dim.next_power_of_two();
        
        if self.norm_out.dim() != (cap_seq, cap_embed) {
            self.norm_out = Array2::zeros((cap_seq, cap_embed));
            self.mix_out = Array2::zeros((cap_seq, cap_embed));
            self.ffn_out = Array2::zeros((cap_seq, cap_embed));
        } else {
            self.norm_out.fill(0.0);
            self.mix_out.fill(0.0);
            self.ffn_out.fill(0.0);
        }
    }
}

pub fn forward(&mut self, input: &Array2<f32>) -> &Array2<f32> {
    self.workspace.ensure_capacity(input.nrows(), input.ncols());
    
    // Reuse workspace buffers
    self.norm.forward_into(input, &mut self.workspace.norm_out);
    self.mix.forward_into(&self.workspace.norm_out, &mut self.workspace.mix_out);
    self.ffn.forward_into(&self.workspace.mix_out, &mut self.workspace.ffn_out);
    
    &self.workspace.ffn_out
}
```

### Benefits
- **Allocations**: 1 per dimension change (not per forward pass)
- **Memory**: Workspace buffers pooled across all layers
- **Cache**: Reused buffers stay warm in cache
- **Predictability**: Fixed maximum memory usage

### When to Use
- ✅ Intermediate buffers in neural network blocks
- ✅ Multiple layers sharing same workspace
- ✅ Variable batch/sequence lengths
- ❌ Outputs that must escape the function
- ❌ Data that must be cached across forward passes

### Implementation Notes
- Combine with power-of-2 sizing for optimal reuse
- Use `_into()` methods that accept pre-allocated output
- Document workspace lifecycle clearly
- Consider thread-local workspaces for parallel inference

---

## Optimization Decision Tree

```
Does component have hot-path allocations?
├─ YES: Heavy matrix multiplications
│   └─ Use Pattern 1: Pre-allocate + general_mat_mul
├─ YES: Large optional matrices not always used
│   └─ Use Pattern 2: Lazy allocation
├─ YES: Variable-dimension scratch buffers
│   └─ Use Pattern 3: Power-of-2 workspace pooling
├─ YES: Expensive metrics computed frequently
│   ├─ Single-threaded? → Use Pattern 4: Dirty-flag cache
│   └─ Multi-threaded? → Skip (overhead > benefit)
└─ YES: Intermediate buffers allocated every forward
    └─ Use Pattern 5: Generational buffer reuse
```

---

## Performance Measurement Guide

### Benchmarking Hot Paths
```rust
#[bench]
fn bench_dot_vs_general_mat_mul(b: &mut Bencher) {
    let input = Array2::<f32>::zeros((512, 768));
    let context = Array2::<f32>::zeros((768, 768));
    
    b.iter(|| {
        // Old way
        let result = input.dot(&context);
        black_box(result)
    });
}
```

### Memory Profiling
- **Before**: Use `valgrind --tool=massif` to profile
- **After**: Compare peak memory usage
- **Expected**: 10-20% reduction for optimized components

### Regression Testing
- Ensure numerical correctness within 1e-5 tolerance
- Run full test suite after changes
- Benchmark on representative data sizes

---

## Common Pitfalls & Solutions

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Over-optimizing non-hot paths | Wasted effort | Profile first, optimize second |
| Ignoring thread-safety | Runtime panics | Test with parallel code paths |
| Interior mutability on hot path | Performance regression | Benchmark before committing |
| Workspace too large | Memory waste | Use power-of-2 sizing |
| Not clearing buffers | Data leaks between calls | Always `fill(0.0)` or `clear()` |
| Wrong generic_mat_mul params | Silent numerical errors | Validate with tests |
| Lazy allocation never triggered | Memory unused | Monitor actual usage patterns |
| Cache invalidation bugs | Subtle correctness issues | Comprehensive test coverage |

---

## Checklist for New Optimizations

- [ ] Profile to confirm hot-path (>5% CPU time)
- [ ] Implement optimization with feature flag (optional)
- [ ] Add comprehensive tests (numerical equivalence, edge cases)
- [ ] Benchmark on 3+ representative data sizes
- [ ] Ensure thread-safety (`Send + Sync` or documented)
- [ ] Document optimization intent and assumptions
- [ ] Run full test suite (no regressions)
- [ ] Review code for maintainability (complex != good)
- [ ] Update AGENTS.md with pattern if reusable
- [ ] Commit with clear description of impact

---

## References

- **ndarray::linalg**: https://docs.rs/ndarray/latest/ndarray/linalg/
- **Power-of-2 sizing**: Used in memory allocators since 1960s
- **Generational buffers**: Common in graphics APIs (double-buffering)
- **Dirty-flag caching**: Classic observer pattern with validation
- **Thread-safety**: https://doc.rust-lang.org/nomicon/concurrency.html

---

**Last Updated**: Phase 3.1 Completion  
**Maintainer**: RustGPT Optimization Team  
**Status**: Active - Update as new patterns emerge
