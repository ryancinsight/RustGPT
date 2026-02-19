# Optimization Quick Reference - Shared Components

## Pattern: Pre-Allocated Matrix Multiplication

### When to Use
Any matrix-vector or matrix-matrix multiplication in hot paths (called every forward/backward pass)

### Before (❌ Allocates)
```rust
let result = matrix.dot(&vector);  // Allocates new Array1
let result = matrix1.dot(&matrix2); // Allocates new Array2
```

### After (✅ No Allocation)
```rust
// For matrix-vector: A (m×n) · v (n×1) = y (m×1)
let mut result = Array1::zeros(matrix.nrows());
{
    let result_len = result.len();
    let vec_len = vector.len();
    let mut result_2d = result.view_mut().into_shape_with_order((result_len, 1))?;
    let vec_2d = vector.view().into_shape_with_order((vec_len, 1))?;
    general_mat_mul(1.0, &matrix, &vec_2d, 0.0, &mut result_2d);
}

// For matrix-matrix: A (m×n) · B (n×p) = C (m×p)
let mut result = Array2::zeros((matrix1.nrows(), matrix2.ncols()));
general_mat_mul(1.0, &matrix1, &matrix2, 0.0, &mut result);
```

### Import Required
```rust
use ndarray::linalg::general_mat_mul;
```

---

## Pattern: Shape Conversion Without Allocation

### Problem
Can't reshape 1D array into 2D for matrix multiplication

### Solution
Use view + `into_shape_with_order()` (creates view, not copy)

```rust
// Extract lengths BEFORE view to avoid borrow conflicts
let arr_len = arr.len();
let mut arr_2d = arr.view_mut().into_shape_with_order((arr_len, 1))?;
// Now use arr_2d without issues

// Read-only version
let arr_len = arr.len();
let arr_2d = arr.view().into_shape_with_order((arr_len, 1))?;
// Now use arr_2d
```

### Key Rules
1. ✅ Extract `.len()` BEFORE calling `.view()`
2. ✅ Extract `.ncols()`, `.nrows()` separately
3. ❌ DON'T call `.len()` within `into_shape_with_order()` call
4. ❌ DON'T borrow same variable mutably and immutably simultaneously

---

## Pattern: Workspace Pooling

### When to Use
Multiple layers need similar scratch buffers (e.g., AdaptiveResiduals)

### Setup (One-Time)
```rust
// In model initialization
pub struct LLMModel {
    pub workspace: Arc<Mutex<AdaptiveResidualsWorkspace>>,
    pub layers: Vec<AdaptiveResiduals>,
}

impl LLMModel {
    pub fn new(...) -> Self {
        let workspace = Arc::new(Mutex::new(AdaptiveResidualsWorkspace::new()));
        
        let layers = (0..num_layers)
            .map(|_| {
                let mut layer = AdaptiveResiduals::new(config);
                layer.set_workspace(Some(workspace.lock().unwrap().clone()));
                layer
            })
            .collect();
        
        Self { workspace, layers }
    }
}
```

### Usage (Per-Layer)
```rust
// Inside layer's forward/backward
impl AdaptiveResiduals {
    pub fn forward(&mut self, input: &Array2<f32>) {
        if let Some(ws) = &mut self.workspace {
            ws.resize_for_dim(embed_dim);  // One-time allocation per dimension change
            // Use ws.nx, ws.mean_x, ws.channel_scales, etc.
        }
    }
}
```

### Benefits
- **Memory**: O(1) workspace, shared across N layers (not O(N))
- **Allocation**: Single resize per dimension change (not per forward pass)
- **Cache**: Larger buffer → better CPU cache utilization

### Power-of-2 Sizing
```rust
// Built into AdaptiveResidualsWorkspace::resize_for_dim()
let new_capacity = embed_dim.next_power_of_two().max(32);

// Example:
// embed_dim=64  → allocate 64
// embed_dim=100 → allocate 128  (avoids realloc if grows to 110)
// embed_dim=200 → allocate 256  (avoids realloc if grows to 250)
```

---

## Pattern: Lazy Allocation

### When to Use
Large buffers that are conditionally used (e.g., similarity context)

### Pattern
```rust
pub struct Component {
    /// Lazily allocated - only created when needed
    expensive_buffer: Option<Array2<f32>>,
}

impl Component {
    pub fn update(&mut self, input: &Array2<f32>) {
        if !self.needs_update() {
            return;
        }
        
        // Allocate only when shape changes
        if self.expensive_buffer.is_none() 
            || self.expensive_buffer.as_ref().unwrap().shape() != [dim, dim] {
            self.expensive_buffer = Some(Array2::zeros((dim, dim)));
        }
        
        // Now use safely
        let buffer = self.expensive_buffer.as_mut().unwrap();
        // ...
    }
}
```

### Benefits
- **Memory**: -2.36 MB per layer (embed_dim 768) when disabled
- **Startup**: Faster initialization
- **Flexibility**: Enable/disable features without memory cost

---

## Pattern: In-Place Operations with Zip

### When to Use
Element-wise operations that can modify in-place

### Pattern: Scalar Operations
```rust
use ndarray::Zip;

// Instead of:
let result = lhs.clone() + &rhs;  // Allocates copy

// Use:
let mut result = lhs.clone();
Zip::from(&mut result)
    .and(&rhs)
    .for_each(|a, &b| *a += b);

// Or for parallel:
Zip::from(&mut result)
    .and(&rhs)
    .par_for_each(|a, &b| *a += b);
```

### Pattern: Row-Wise Operations
```rust
// Instead of:
for row in output.outer_iter_mut() {
    row += &scaling_vector;
}

// Parallel version (if elements >= PARALLEL_MIN):
Zip::from(output.outer_iter_mut())
    .par_for_each(|mut row| {
        row += &scaling_vector;
    });
```

### When to Parallelize
```rust
const PARALLEL_MIN_ELEMENTS: usize = 4_096;

if output.len() >= PARALLEL_MIN_ELEMENTS && output.nrows() > 1 {
    // Use .par_for_each()
} else {
    // Use sequential loop
}
```

---

## Performance Guidelines

### Allocation-Free Hot Path Checklist
- [ ] All `.dot()` calls replaced with `general_mat_mul`
- [ ] Output buffers pre-allocated
- [ ] No `.clone()` in loop (unless unavoidable)
- [ ] `Zip::par_for_each()` for parallel data access
- [ ] Workspace reused across layers (if applicable)
- [ ] Lazy allocation for conditional features

### Memory Savings
| Optimization | Per-Layer | Per-12-Layer Model |
|--------------|-----------|-------------------|
| general_mat_mul | 5-50 KB | 60-600 KB |
| Workspace pooling | 50-100 KB | 50-100 KB (total) |
| Lazy allocation | 2-10 KB | 24-120 KB |
| In-place ops | 10-50 KB | 120-600 KB |
| **TOTAL** | **65-210 KB** | **254-1420 KB** |

### Benchmark Commands
```bash
# Build optimized
cargo build --release

# Run specific component tests
cargo test --lib conditioning --release

# Profile allocations (with valgrind on Linux)
valgrind --tool=massif --massif-out-file=massif.out ./target/release/llm_test
ms_print massif.out | head -50
```

---

## Migration Checklist

When optimizing a new component:

- [ ] Identify all `.dot()` calls in hot paths
- [ ] Pre-allocate output buffers
- [ ] Use `general_mat_mul(1.0, &A, &B, 0.0, &mut C)`
- [ ] Test with `cargo check` first
- [ ] Fix borrow checker by extracting lengths
- [ ] Format with `cargo fmt`
- [ ] Run tests: `cargo test --lib <component_name>`
- [ ] Update documentation with memory impact
- [ ] Create optimization summary doc

---

## Common Mistakes

### ❌ DON'T: Shape convert with `.len()` inside call
```rust
let mut arr_2d = arr.view_mut().into_shape_with_order((arr.len(), 1))?;
// ERROR: arr is borrowed mutably and immutably simultaneously
```

### ✅ DO: Extract length first
```rust
let arr_len = arr.len();
let mut arr_2d = arr.view_mut().into_shape_with_order((arr_len, 1))?;
// OK: length extracted before view
```

### ❌ DON'T: Reallocate workspace every forward pass
```rust
pub fn forward(&mut self) {
    self.workspace = AdaptiveResidualsWorkspace::new();  // WRONG!
}
```

### ✅ DO: Resize (not reallocate) only when needed
```rust
pub fn forward(&mut self) {
    if let Some(ws) = &mut self.workspace {
        ws.resize_for_dim(embed_dim);  // Only reallocates if needed
    }
}
```

### ❌ DON'T: Use sequential Zip for large tensors
```rust
Zip::from(&mut arr).and(&other).for_each(|a, &b| *a += b);
// Slow on multi-core
```

### ✅ DO: Use parallel Zip when possible
```rust
if arr.len() >= 4096 {
    Zip::from(&mut arr).and(&other).par_for_each(|a, &b| *a += b);
} else {
    Zip::from(&mut arr).and(&other).for_each(|a, &b| *a += b);
}
```

---

## References
- **Thread**: @T-019c54ca-de8b-770a-9f4b-b0fa11cd1f72
- **Pattern Examples**: `src/domain/layers/components/`
  - `attention_context.rs` - general_mat_mul reference
  - `adaptive_residuals.rs` - workspace pooling reference
  - `conditioning.rs` - complete optimization example
- **ndarray Docs**: https://docs.rs/ndarray/
