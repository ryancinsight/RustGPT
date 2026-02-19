# Quick Reference: SharedFeedforward GPU Wiring (Phase 5.6)

## 1-Minute Summary

✅ **Richards Activation GPU Kernels**: Implemented in `richards_glu_fused_kernel.rs`
- Function: `apply_richards_activation_gpu()` dispatches to `GpuDevice::richards_curve()`
- Zero intermediate transfers: All computation stays on GPU
- Backend-agnostic: Works with CUDA, Metal, WGPU

✅ **SharedFeedforward Integration**: Already complete
- Calls `FeedForwardVariant::forward_gpu()` which routes to `RichardsGlu::forward_gpu()`
- GPU device is stored as `Option<Arc<Mutex<GpuDevice>>>`
- Strict no-fallback: Returns error if GPU unavailable

---

## Code Flow: Input → GPU Forward → Output

```
User Code (e.g., inference loop)
    ↓
SharedFeedforward::forward(&input)
    ↓
Check compute_backend.is_gpu() ?
    ├─ YES → forward_gpu(&input)
    │   ↓
    │   FeedForwardVariant::forward_gpu(&input)
    │   ↓
    │   RichardsGlu::forward_gpu(&input)
    │   ├─ Lock GPU device
    │   ├─ Upload input to GPU
    │   ├─ Call forward_gpu_kernel()
    │   │   ├─ GEMM: input @ w1 → x1
    │   │   ├─ GEMM: input @ w2 → x2
    │   │   ├─ apply_richards_activation_gpu()
    │   │   │   ├─ GPU Kernel: richards_curve(x1) → sigma
    │   │   │   ├─ GPU Kernel: mul(x1 * sigma) → value
    │   │   │   └─ GPU Kernel: richards_curve(x2) → gate
    │   │   ├─ GPU Kernel: mul(value * gate) → gated
    │   │   └─ GEMM: gated @ w_out → output
    │   ├─ Download output from GPU
    │   └─ Deallocate intermediate buffers
    │   ↓
    │   Return output
    │
    └─ NO → forward(&input)  [CPU path]
        ↓
        Return output
```

---

## Key Functions

### 1. `apply_richards_activation_gpu()` (NEW)

**Location**: `src/domain/compute/richards_glu_fused_kernel.rs:203`

**Purpose**: Dispatch Richards activation to GPU kernels without CPU fallback.

**Signature**:
```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn apply_richards_activation_gpu(
    device: &mut GpuDevice,
    x1: &GpuBuffer,
    x2: &GpuBuffer,
    value: &mut GpuBuffer,
    gate: &mut GpuBuffer,
    batch_size: usize,
    hidden_dim: usize,
    params: &OptimizedRichardsGluParams,
) -> Result<()>
```

**GPU Kernels Called**:
1. `device.richards_curve(&x1, &mut sigma, &value_params, total_size)` - Activation
2. `device.mul(&x1, &sigma, &mut value, total_size)` - Multiply by input
3. `device.richards_curve(&x2, &mut gate, &gate_params, total_size)` - Gate activation

**Memory**: Temporary buffer `sigma` allocated and deallocated within function.

---

### 2. `forward_gpu()` (MODIFIED)

**Location**: `src/domain/compute/richards_glu_fused_kernel.rs:275`

**What Changed**: 
- **Before**: Downloaded x1, x2 to CPU, computed activation on CPU, uploaded results
- **After**: Calls `apply_richards_activation_gpu()` to keep computation on GPU

**Before (CPU Fallback)**:
```rust
device.download(&x1, &mut x1_cpu)?;
device.download(&x2, &mut x2_cpu)?;
for i in 0..batch_size {
    for j in 0..hidden_dim {
        let sigma = richards_activation(x1_cpu[idx], params);
        value_cpu[idx] = x1_cpu[idx] * sigma;
        // ... CPU computation ...
    }
}
device.upload(&value_cpu, &mut value)?;
device.upload(&gate_cpu, &mut gate)?;
```

**After (GPU Dispatch)**:
```rust
apply_richards_activation_gpu(
    device,
    &x1,
    &x2,
    &mut value,
    &mut gate,
    batch_size,
    hidden_dim,
    params,
)?;
```

---

### 3. `SharedFeedforward::forward_gpu()` (EXISTING)

**Location**: `src/domain/layers/components/feedforward.rs:186`

**Purpose**: Entry point for GPU-accelerated feedforward from SharedFeedforward.

**Signature**:
```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    self.feedforward.ensure_gpu_device_auto_detect()?;
    self.feedforward.forward_gpu(input)
}
```

**Flow**:
1. Ensure GPU device is attached (strict auto-detect, errors if no GPU)
2. Delegate to `FeedForwardVariant::forward_gpu()`
3. Returns `Result<Array2<f32>>` (GPU output as CPU array)

---

### 4. `GpuDevice::backend_name()` (NEW)

**Location**: `src/domain/compute/gpu_device.rs:196`

**Purpose**: Expose backend name for kernel dispatch decisions.

**Signature**:
```rust
pub fn backend_name(&self) -> &'static str {
    self.backend.as_str()
}
```

**Returns**: `"cuda"`, `"metal"`, or `"vulkan"` (for WGPU)

---

## Parameter Conversion

### OptimizedRichardsGluParams → RichardsCurveParams

**In** `apply_richards_activation_gpu()`:

```rust
// Value activation parameters (for x1)
let value_params = crate::domain::compute::gpu_ops::RichardsCurveParams {
    nu: params.nu,                              // Center
    k: params.k,                                // Steepness
    m: params.m,                                // Shape
    beta: params.beta,                          // Exponent scale
    temperature: params.temp_reciprocal.recip(), // Reciprocal of reciprocal = original
};

// Gate activation parameters (for x2)
let gate_params = crate::domain::compute::gpu_ops::RichardsCurveParams {
    nu: 0.0,                                    // Gate centered at 0
    k: params.gate_scale,                       // Gate steepness
    m: 1.0,                                     // Simple curve
    beta: params.gate_bias,                     // Gate exponent scale
    temperature: params.gate_temp_reciprocal.recip(),
};
```

---

## Error Handling

### Strict No-Fallback Policy

**SharedFeedforward**:
```rust
if self.compute_backend.is_gpu() {
    return self.forward_gpu(input).expect("GPU forward pass failed");
}
```

If GPU is selected but unavailable → **PANIC** (not silent fallback)

**Rationale**: Predictable performance. No surprise CPU computation on GPU systems.

---

## Memory Lifecycle

### Buffer Allocation & Deallocation

```rust
// Allocated in forward_gpu()
let mut x1 = device.allocate_f32(x1_size)?;
let mut x2 = device.allocate_f32(x2_size)?;
let mut value = device.allocate_f32(hidden_size)?;
let mut gate = device.allocate_f32(hidden_size)?;
let mut gated = device.allocate_f32(hidden_size)?;

// Passed to apply_richards_activation_gpu()
// Temporary sigma buffer allocated inside apply_richards_activation_gpu()

// Deallocated in forward_gpu()
device.deallocate(x1);
device.deallocate(x2);
device.deallocate(value);
device.deallocate(gate);
device.deallocate(gated);
```

**Total Allocations**: 1 input + 4 intermediate + 1 output = 6 buffers
**Total Deallocations**: Same 6 buffers

---

## Testing

### Enable GPU & Run Tests

```bash
# WGPU (portable)
cargo test --lib --features wgpu

# CUDA (NVIDIA)
cargo test --lib --features gpu-cuda

# Metal (macOS)
cargo test --lib --features gpu-metal
```

### Test Locations

1. **Unit Test**: `richards_glu_fused_kernel.rs::test_gpu_forward_dispatch`
   - Verifies GPU forward dispatch works
   - Tests with small batch (2×64→128→64)
   - Gracefully skips on CPU-only systems

2. **Integration Test**: `feedforward.rs` (to be added)
   - Tests `SharedFeedforward::forward_gpu()` path
   - Verifies numerical accuracy vs CPU (ε ≤ 1e-4)

---

## Performance Checklist

- [x] No intermediate downloads/uploads between GPU kernels
- [x] All GPU launches are sequential (pipelined by GPU scheduler)
- [x] Input/output are the only CPU↔GPU transfers
- [x] Memory buffers are deallocated after use
- [x] Parameter conversion is zero-copy (just field remapping)
- [x] Backend selection is automatic (CUDA > Metal > WGPU)

---

## Compilation Status

**Status**: ✅ PASSING

```bash
$ cargo check --lib
    Checking llm v0.1.0 (D:\RustGPT)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.22s
```

No errors or warnings related to GPU kernel wiring.

---

## Next Priority Items

**Priority 3**: PolyAttention GPU Support
- File: `src/domain/layers/components/poly_attention.rs`
- Task: Implement `GpuComponent` and fused GPU kernel
- Estimated speedup: 30x for 512-batch

**Priority 4**: Mamba Scan GPU Kernel
- File: `src/domain/layers/components/temporal_processing_gpu.rs`
- Task: Implement selective scan on GPU
- Estimated speedup: 20x for 512-batch

**Priority 5**: Code Cleanup
- Remove CPU-only duplicate kernels
- Consolidate backend-specific code
- Add comprehensive GPU benchmarks

---

## Quick Debugging

### GPU Forward Pass Fails

**Check 1**: Is GPU feature enabled?
```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
```

**Check 2**: Is GPU device attached?
```rust
let ff = SharedFeedforward::new(...);
ff.enable_gpu_auto_detect()?;  // Must call this
ff.set_compute_backend_checked(ComputeBackend::Vulkan)?;  // Select backend
```

**Check 3**: Are parameters correctly converted?
```rust
let value_params = RichardsCurveParams {
    temperature: params.temp_reciprocal.recip(),  // Not params.temp_reciprocal
    // ...
};
```

### Numerical Mismatch

**Expected**: Max error ≤ 1e-4 vs CPU
**Reason**: Floating-point rounding in GPU kernels
**Tolerance**: Use `approx::abs_diff_eq!(gpu_out, cpu_out, epsilon = 1e-4)`

---

## References

- **Main Implementation**: `GPU_KERNEL_WIRING_PHASE5_6.md`
- **GPU Device API**: `src/domain/compute/gpu_device.rs`
- **FeedForward Types**: `src/domain/layers/components/common.rs`
- **RichardsGlu GPU**: `src/domain/richards/richards_glu.rs`
