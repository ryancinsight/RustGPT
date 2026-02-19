# Phase 5.6.3: GPU Backend Consolidation - Code Locations & References

**Purpose**: Concrete file paths and line numbers for implementation  
**Status**: Ready for execution

## Key Files to Modify (In Order)

### 1. GPU Operations Trait Definition (FIRST)

**File**: `src/domain/compute/gpu_ops.rs`

**Action**: Add method signature to `GpuMatrixOps` trait

**Location**: Around line 250 (after `moh_gate_activation`)

**What to add**:
```rust
/// Richards GLU Fused Kernel: Two-Pass Gating
fn richards_glu_fused(
    &mut self,
    pool: &mut dyn GpuMemoryPool,
    input: &GpuBuffer,           // [batch, input_dim]
    w_g1: &GpuBuffer,            // [input_dim, hidden_dim]
    w_g2: &GpuBuffer,            // [input_dim, hidden_dim]
    w_out: &GpuBuffer,           // [hidden_dim, output_dim]
    output: &mut GpuBuffer,      // [batch, output_dim]
    batch_size: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> Result<()>;
```

**Also add** to `CpuGpuMatrixOps` impl (around line 599, after `richards_curve`):
```rust
fn richards_glu_fused(...) -> Result<()> {
    Err(ModelError::Backend {
        message: "CPU richards_glu_fused not implemented".to_string(),
    })
}
```

---

### 2. WGPU Implementation (SECOND - CRITICAL PATH)

**File**: `src/domain/compute/wgpu_ops.rs`

**Action**: Implement `richards_glu_fused()` method

**Location**: After `fn sigmoid(...)` implementation, around line 500

**Implementation strategy**:
1. Allocate intermediate buffer `gated_buf` for `[batch, hidden_dim]`
2. Allocate intermediate buffer `value_buf` for `[batch, hidden_dim]`
3. Call `self.gemm_f32()` for `input @ w_g1 → value`
4. Call `self.gemm_f32()` for `input @ w_g2 → gate_logits`
5. Call `self.richards_curve()` on `gate_logits → gate`
6. Call `self.mul()` for `gate * value → gated`
7. Call `self.gemm_f32()` for `gated @ w_out → output`
8. Deallocate `value_buf`, `gate_logits_buf`, `gate_buf` (keep gated if needed)
9. Return `Ok(())`

**See**: `PHASE5.6.3_RICHARDS_GLU_IMPLEMENTATION_GUIDE.md` for detailed code

---

### 3. CUDA Stub Implementation

**File**: `src/domain/compute/cuda/ops.rs`

**Action**: Add error stub pointing to `.cu` file

**Location**: Around line 200+ (in `impl GpuMatrixOps for CudaMatrixOps`)

**What to add**:
```rust
fn richards_glu_fused(
    &mut self,
    _pool: &dyn GpuMemoryPool,
    _input: &GpuBuffer,
    _w_g1: &GpuBuffer,
    _w_g2: &GpuBuffer,
    _w_out: &GpuBuffer,
    _output: &mut GpuBuffer,
    batch_size: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> Result<()> {
    Err(ModelError::Backend {
        message: format!(
            "CUDA richards_glu_fused not yet implemented. \
             See src/domain/compute/cuda/kernels/richards_glu_fused.cu"
        ),
    })
}
```

---

### 4. Metal Stub Implementation

**File**: `src/domain/compute/metal/ops.rs`

**Action**: Add error stub pointing to `.metal` file

**Location**: Similar location as CUDA (around line 200+)

**What to add**: Same as CUDA stub but reference `.metal` file instead

---

## Working Directories

### GPU Compute Backend
```
src/domain/compute/
├── gpu_ops.rs                          (Trait definitions)
├── gpu_device.rs                       (Auto-detection, priority)
├── gpu_memory.rs                       (Memory pool)
├── wgpu_ops.rs                         (WGSL implementations) ✅
├── cuda/
│   ├── ops.rs                          (CUDA stubs)
│   ├── memory.rs                       (CUDA memory pool)
│   └── kernels/                        (NEW - to create)
└── metal/
    ├── ops.rs                          (Metal stubs)
    ├── memory.rs                       (Metal memory pool)
    └── kernels/                        (NEW - to create)
```

### Unified GPU Kernels
```
src/domain/layers/components/
├── unified_gpu_kernels.rs              (Main dispatcher)
├── unified_gpu_backend.rs              (Backend abstraction)
└── unified_buffer_pool.rs              (Buffer management)
```

### Tests
```
tests/
├── gpu_shared_components_phase56.rs    (Integration tests) ← Run these
├── gpu_backend_detection.rs            (Detection tests)
├── gpu_kernels_phase56_verification.rs (Kernel verification)
└── gpu_richardson_glu_fused.rs         (RichardsGLU tests)
```

---

## WGPU Methods to Reuse

These are already implemented in `src/domain/compute/wgpu_ops.rs`:

| Method | Purpose | Line Range |
|--------|---------|-----------|
| `fn gemm_f32(...)` | Tiled matrix multiplication | ~200 |
| `fn gemv_f32(...)` | Matrix-vector multiply | ~250 |
| `fn relu(...)` | Element-wise ReLU | ~300 |
| `fn gelu(...)` | Element-wise GELU | ~350 |
| `fn silu(...)` | Element-wise SiLU | ~400 |
| `fn mul(...)` | Element-wise multiplication | ~500 |
| `fn add_scaled(...)` | Scaled addition | ~550 |
| `fn scale(...)` | Scale buffer | ~600 |
| `fn axpy(...)` | BLAS axpy | ~650 |
| `fn richards_curve(...)` | Richards sigmoid | ~700 |
| `fn softmax(...)` | Log-sum-exp softmax | ~750 |
| `fn layer_norm(...)` | Layer normalization | ~800 |

**Reuse pattern**:
```rust
// Call existing WGPU methods directly
self.gemm_f32(pool, 1.0, input, w_g1, 0.0, &mut value_buf, ...)?;
self.richards_curve(pool, &gate_logits, &mut gate, &params, ...)?;
self.mul(pool, &gate, &value, &mut gated, ...)?;
```

---

## Build Verification Commands

### Check Compilation
```bash
cargo check --lib --features wgpu
cargo check --lib --features gpu-cuda
cargo check --lib --features gpu-metal
```

### Run Unit Tests
```bash
cargo test --lib gpu_ops --features wgpu
cargo test --lib gpu_device --features wgpu
```

### Run Integration Tests
```bash
cargo test --test gpu_shared_components_phase56 --features wgpu
cargo test --test gpu_backend_detection --features wgpu
```

### Run with All Backends
```bash
cargo test --lib --features gpu-all
cargo test --test gpu_shared_components_phase56 --features gpu-all
```

### Benchmarks
```bash
cargo bench --bench gpu_performance --features wgpu -- richards_glu
cargo bench --bench gpu_performance --features gpu-cuda -- richards_glu
```

---

## Execution Steps (Concrete)

### Step 1: Update GPU Ops Trait
```bash
# Edit src/domain/compute/gpu_ops.rs
# Find: pub trait GpuMatrixOps: Send + Sync {
# Add: fn richards_glu_fused(...) -> Result<()>;
# Around line 250 (after moh_gate_activation)

cargo check --lib --features wgpu
```

### Step 2: Implement in WGPU
```bash
# Edit src/domain/compute/wgpu_ops.rs
# Find: impl GpuMatrixOps for WgpuMatrixOps { ... }
# Add: fn richards_glu_fused(...) -> Result<()> { ... }
# Around line 500 (after sigmoid implementation)

cargo check --lib --features wgpu
```

### Step 3: Add CUDA Stub
```bash
# Edit src/domain/compute/cuda/ops.rs
# Find: impl GpuMatrixOps for CudaMatrixOps { ... }
# Add: fn richards_glu_fused(...) -> Result<()> { ... }
# Return Err with informative message

cargo check --lib --features gpu-cuda
```

### Step 4: Add Metal Stub
```bash
# Edit src/domain/compute/metal/ops.rs
# Similar to CUDA stub

cargo check --lib --features gpu-metal
```

### Step 5: Create CUDA Kernel Templates
```bash
# Create directory: src/domain/compute/cuda/kernels/
# Create files:
#   - element_wise.cu (template)
#   - richards_curve.cu (template)
#   - richards_glu_fused.cu (template)
#   - activation.cu (template)
#   - Makefile.cu (build script template)

cargo check --lib --features gpu-cuda
```

### Step 6: Test & Verify
```bash
cargo test --test gpu_shared_components_phase56 --features wgpu
cargo bench --bench gpu_performance --features wgpu -- richards_glu
```

---

## Git Commands

```bash
# Check status
git status

# Stage changes
git add src/domain/compute/
git add src/domain/compute/cuda/kernels/
git add PHASE5.6.3_*.md

# Commit
git commit -m "Phase 5.6.3: GPU backend consolidation - RichardsGLU fused kernel (WGPU)"

# Push
git push origin main

# Or create feature branch
git checkout -b phase/5.6.3/gpu-consolidation
git push origin phase/5.6.3/gpu-consolidation
```

---

## Documentation Reference

### Planning Documents
1. **PHASE5.6.3_QUICKSTART.md** - Build commands & quick reference
2. **SESSION_KICKOFF_PHASE5.6.3.md** - Hour-by-hour execution plan
3. **PHASE5.6.3_IMPLEMENTATION_PLAN.md** - Detailed roadmap
4. **PHASE5.6.3_GPU_KERNEL_STATUS.md** - Kernel inventory
5. **PHASE5.6.3_RICHARDS_GLU_IMPLEMENTATION_GUIDE.md** - Use during coding
6. **PHASE5.6.3_DOCUMENTATION_INDEX.md** - Master navigation
7. **THREAD_SUMMARY_PHASE5.6.3_KICKOFF.md** - Session summary
8. **PHASE5.6.3_CODE_LOCATIONS.md** - This file

### Read Order
1. SESSION_KICKOFF_PHASE5.6.3.md (5 min overview)
2. PHASE5.6.3_QUICKSTART.md (quick reference)
3. PHASE5.6.3_RICHARDS_GLU_IMPLEMENTATION_GUIDE.md (during coding)
4. This file (concrete line numbers)

---

## Expected Timeline

| Hour | Activity | File Changes |
|------|----------|--------------|
| 1 | Verify build, plan | None yet |
| 2 | Update trait + WGPU | gpu_ops.rs, wgpu_ops.rs |
| 3 | Add CUDA/Metal stubs | cuda/ops.rs, metal/ops.rs |
| 4 | Test & benchmark | Tests passing |
| 5 | CUDA templates | cuda/kernels/*.cu |

---

## Success Checklist

- [ ] `cargo check --lib` passes
- [ ] gpu_ops.rs trait updated
- [ ] wgpu_ops.rs implementation complete
- [ ] CUDA/Metal stubs added
- [ ] `cargo test --features wgpu` passes
- [ ] `cargo test --test gpu_shared_components_phase56` passes
- [ ] Benchmark shows 2 kernel launches (not 5+)
- [ ] CUDA kernel templates created

---

Ready to implement. Start with Step 1 above.
