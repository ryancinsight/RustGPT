# GPU Backend Implementation Guide (Phase 5.6.3)
## Complete Reference for Kernel Integration & Optimization

**Status**: Implementation Ready  
**Date**: 2026-02-16

---

## 1. Architecture Overview

### Unified GPU Backend Stack

```
┌─────────────────────────────────────────────────────────────────┐
│ High-Level API: UnifiedGpuBackend                               │
│ - auto_detect() for GPU selection                               │
│ - Strict no-fallback semantics                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────v──────────────────┐  ┌──v──────────────────────────┐
│ UnifiedGpuKernels        │  │ Shared Components           │
│ (Kernel Dispatcher)      │  │ - Attention                 │
│ - Workspace Management   │  │ - Feedforward (RichardsGLU) │
│ - Fused Kernels          │  │ - Temporal (Mamba/RG-LRU)   │
└───────┬──────────────────┘  └──┬──────────────────────────┘
        │                        │
        └────────────┬───────────┘
                     │
        ┌────────────v────────────┐
        │  GpuDevice              │
        │  (Device Management)    │
        └────────┬────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
   ┌────v─────┐  ┌────────v───┐
   │ WGPU Ops │  │ CUDA/Metal │
   │ (Primary)│  │ (Fast Path)│
   └──────────┘  └────────────┘
```

### Trait Hierarchy

```rust
GpuMatrixOps (trait)
├── WgpuMatrixOps (impl for WGPU)
├── CudaMatrixOps (impl for CUDA)
└── MetalMatrixOps (impl for Metal)

GpuMemoryPool (trait)
├── WgpuMemoryPool (impl for WGPU)
├── CudaMemoryPool (impl for CUDA)
└── MetalMemoryPool (impl for Metal)
```

---

## 2. Kernel Implementation Patterns

### Pattern 1: Element-wise Operations

**Location**: `src/domain/compute/wgsl_kernels.rs`

```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    // Element-wise computation
    output[idx] = activate(input[idx]);
}
```

**Implementation Dispatch**:
```rust
// In wgpu_ops.rs
impl GpuMatrixOps for WgpuMatrixOps {
    fn richards_curve(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &RichardsCurveParams,
        size: usize,
    ) -> Result<()> {
        // 1. Create bind group with input, output, params
        // 2. Create compute pipeline from WGSL shader
        // 3. Dispatch workgroups: (size + 255) / 256
        // 4. Wait for completion
        Ok(())
    }
}
```

### Pattern 2: Reduction Operations

**Location**: `src/domain/compute/wgsl_kernels.rs` - `SHADER_SUM`

```wgsl
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let idx = global_id.x;
    
    // 1. Load into shared memory
    shared_sum[tid] = input[idx];
    workgroupBarrier();
    
    // 2. Tree reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        workgroupBarrier();
    }
    
    // 3. Write result
    if (tid == 0u) {
        output[0] = shared_sum[0];
    }
}
```

### Pattern 3: Fused Operations

**Location**: `src/domain/compute/wgsl_kernels.rs` - `SHADER_RICHARDS_GLU_PASS1`

```wgsl
// Combine multiple operations in single kernel to minimize global memory traffic

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Read once: x1[idx], x2[idx]
    let x1_val = x1[idx];
    let x2_val = x2[idx];
    
    // Compute multiple outputs from single read
    let sigma1 = richards_activation(x1_val);
    value[idx] = x1_val * sigma1;        // Output 1
    
    gate[idx] = gate_activation(x2_val);  // Output 2
    
    // Single write transaction for multiple results
}
```

---

## 3. Workspace Memory Management

### Buffer Lifecycle

```rust
// 1. ALLOCATION (ensure_capacity)
let mut kernels = UnifiedGpuKernels::auto_detect()?;
kernels.ensure_capacity(batch_size, embed_dim, seq_len)?;
// → Allocates power-of-2 sized buffers in GpuKernelWorkspace
// → Buffers stored in Vec<GpuBuffer> for reuse

// 2. REUSE (multiple operations)
kernels.attention_forward(...)?;   // Reuses workspace buffers
kernels.feedforward_forward(...)?; // Reuses same buffers
// → No deallocations; buffers overwritten in place

// 3. RESET (optional, between different batch shapes)
kernels.reset_workspace();
// → Marks buffers ready for reuse
// → Checks if capacity sufficient for new dimensions

// 4. CLEANUP (end of session)
kernels.cleanup_workspace()?;
// → Deallocates all workspace buffers
// → Resets capacity tracking
```

### Power-of-2 Sizing Strategy

```rust
// Example: batch_size=33, embed_dim=769, seq_len=128

// BEFORE: Power-of-2 sizing
let new_batch = 33.next_power_of_two();  // 64
let new_embed = 769.next_power_of_two(); // 1024
let new_seq = 128.next_power_of_two();   // 128

// BENEFIT:
// - 64 is multiple of workgroup size 256 (partial, but efficient)
// - 1024 is multiple of matrix tile sizes (16×16 = 256)
// - Coalesced memory access patterns align to cache lines
// - Reduction in bank conflicts in shared memory
```

### Buffer Pool Structure

```rust
struct GpuKernelWorkspace {
    capacity: (usize, usize, usize),           // Current capacity
    buffers: Vec<GpuBuffer>,                   // Allocated buffers
    ready: bool,                               // Ready for use
    allocation_count: usize,                   // Tracking stats
    reallocation_count: usize,                 // How many times resized
}

// Buffer allocation order:
// [0] activation_0     [batch×embed] – intermediate activations
// [1] activation_1     [batch×embed] – intermediate activations
// [2] qkv_0            [batch×embed] – query/key/value
// [3] qkv_1            [batch×embed] – query/key/value
// [4] qkv_2            [batch×embed] – query/key/value
// [5] scores           [batch×seq×seq] – attention scores
// [6] attn_output      [batch×embed] – attention output
// [7] weight           [embed×embed] – weight matrices
```

---

## 4. Strict No-Fallback Error Handling

### Error Propagation Design

All GPU operations return explicit errors instead of silent CPU fallback:

```rust
// ✅ CORRECT: Explicit GPU-only operation
pub fn richards_curve(
    &mut self,
    pool: &mut dyn GpuMemoryPool,
    input: &GpuBuffer,
    output: &mut GpuBuffer,
    params: &RichardsCurveParams,
    size: usize,
) -> Result<()> {
    // Returns Err if GPU not available or kernel fails
}

// ❌ WRONG: Silent CPU fallback (forbidden)
pub fn richards_curve(...) -> Result<()> {
    match device.run_kernel(...) {
        Ok(()) => Ok(()),
        Err(_) => {
            // DON'T DO THIS: compute on CPU and return Ok
            cpu_richards_curve(...);
            Ok(())
        }
    }
}
```

### Error Categories

```rust
ModelError::Backend {
    message: String,
}

// Error Messages

// 1. GPU NOT AVAILABLE
"Automatic GPU detection failed: no supported GPU backend detected (CUDA/Metal/Vulkan)"

// 2. MISSING IMPLEMENTATION
"GPU operation 'richards_curve' not implemented for CUDA backend. \
 Implementation stub ready at src/domain/compute/kernels/richards_curve.cu"

// 3. COMPILATION FAILURE
"WGPU shader compilation failed for 'SHADER_RICHARDS_CURVE': {details}"

// 4. MEMORY ALLOCATION FAILURE
"GPU memory allocation failed: requested 4GB but only 2GB available"

// 5. EXECUTION FAILURE
"GPU kernel execution error: invalid dimensions for kernel dispatch"
```

---

## 5. Richards Curve Kernel Implementation

### Mathematical Definition

```
Richards(x) = 1 / (1 + (k*m)^(1/m) * exp(-β*(x-ν)))

Parameters:
- ν (nu):    Inflection point / center
- k:         Growth rate / steepness
- m:         Shape parameter (asymmetry)
- β (beta):  Scale / temperature
```

### Numerical Stability

```wgsl
// Prevent overflow/underflow in exp()

let exponent = -params.beta * center;

// Case 1: Exponent too large (exp → ∞)
if (exponent > 20.0) {
    exp_val = 1e38;  // Approximate ∞
    // Result: σ ≈ 1 / (1 + 1e38) ≈ 0
}

// Case 2: Exponent too small (exp → 0)
else if (exponent < -20.0) {
    exp_val = 1e-38;  // Approximate 0
    // Result: σ ≈ 1 / (1 + 0) ≈ 1
}

// Case 3: Normal range
else {
    exp_val = exp(exponent);
}

// Final computation with epsilon for numerical safety
let sigma = 1.0 / (denominator + 1e-8);
```

### Per-Head Gating (MoH)

For PolyAttention gates with per-head parameters:

```wgsl
// MoH Gate Activation
let scaled_logit = logits[idx] * alpha[head_idx] + beta_vals[head_idx];
output[idx] = richards_activation(scaled_logit);

// Each head can have different (α, β) parameters for fine-grained control
```

---

## 6. Performance Tuning Guide

### Workgroup Size Selection

| Operation Type | Workgroup Size | Rationale |
|---|---|---|
| Element-wise | 256 | Maximum occupancy, hide memory latency |
| Matrix multiply | 16×16 = 256 | Tile size matches shared memory pattern |
| Reduction | 256 | Full-width reduction with shared memory |
| Softmax | 256 | Row-wise reduction across full row |

### Memory Access Patterns

```rust
// ✅ GOOD: Coalesced linear access
for (var i = tid; i < total; i += 256) {
    output[i] = input[i] * scale;  // Sequential thread access
}

// ❌ BAD: Strided/random access
for (var i = 0; i < total; i += 1) {
    let idx = (i + tid * 17) % total;  // Random indices
    output[idx] = input[idx] * scale;  // Cache misses
}
```

### Shared Memory Usage

```wgsl
// Keep shared memory allocations < 48KB per workgroup
// Example: 256 threads × f32 (4 bytes) = 1KB
var<workgroup> shared_sum: array<f32, 256>;    // 1KB ✅
var<workgroup> shared_data: array<f32, 8192>;  // 32KB ✅
var<workgroup> shared_huge: array<f32, 16384>; // 64KB ❌ (exceeds limit)
```

### Kernel Launch Overhead

```rust
// Minimize kernel launches:
// - Single fused kernel: 1 launch
// - Multiple separate kernels: N launches + overhead

// RichardsGLU Performance:
// Option 1 (5 launches):
//   1. x1 = input @ w1
//   2. x2 = input @ w2
//   3. value = x1 * richards(x1)
//   4. gate = richards(x2)
//   5. output = gated @ w_out
// Cost: 5 launches × overhead

// Option 2 (2 launches, fused):
//   1. Pass 1: [x1, x2] → [value, gate]
//   2. Pass 2: [gated] → output
// Cost: 2 launches × overhead + zero memory traffic
// Speedup: ~2.5x from reduced overhead alone
```

---

## 7. Automatic GPU Detection Implementation

### Detection Priority Order

```rust
// src/domain/compute_backend.rs

pub fn detect_available_gpu_backends() -> Vec<ComputeBackend> {
    let mut backends = Vec::new();
    
    // Priority 1: CUDA (highest performance on NVIDIA)
    if has_cuda_compiler && cuda_device_found {
        backends.push(ComputeBackend::Cuda);
    }
    
    // Priority 2: Metal (native on macOS)
    if cfg!(target_os = "macos") && metal_device_found {
        backends.push(ComputeBackend::Metal);
    }
    
    // Priority 3: Vulkan/WGPU (universal fallback)
    if wgpu_supported && vulkan_device_found {
        backends.push(ComputeBackend::Vulkan);
    }
    
    backends
}
```

### Feature Flag Integration

```toml
# Cargo.toml features

[features]
gpu-cuda = ["cudarc", "proc-macro2"]
gpu-metal = ["metal", "objc"]
wgpu = ["wgpu", "wgsl"]
gpu-all = ["gpu-cuda", "gpu-metal", "wgpu"]
```

### Runtime Detection

```rust
pub fn auto_detect() -> Result<GpuDevice> {
    // 1. Check compile-time enabled features
    let detected_backends = detect_available_gpu_backends();
    
    // 2. Check runtime device availability
    let runtime_backends = detect_available_gpu_backends_runtime();
    
    // 3. Select highest-priority available backend
    if let Some(backend) = detected_backends.first() {
        GpuDevice::new(*backend)
    } else if !runtime_backends.is_empty() {
        // GPU found but binary not compiled with matching features
        Err("GPU detected but binary missing feature flags")
    } else {
        // No GPU found
        Err("No GPU detected on this system")
    }
}
```

---

## 8. Testing Strategy

### Unit Tests (Kernel Correctness)

```rust
#[test]
fn test_richards_curve_bounds() {
    // Test: σ(x) ∈ (0, 1) for all x
    // Test: σ is monotonically increasing
    // Test: σ(ν) ≈ 0.5 (inflection at center)
}

#[test]
fn test_workspace_capacity_growth() {
    // Test: Power-of-2 sizing
    // Test: Reallocation only when needed
    // Test: Buffers reused across calls
}

#[test]
#[cfg(feature = "wgpu")]
fn test_wgpu_kernel_dispatch() {
    // Test: Kernel compiles successfully
    // Test: Correct bind group layout
    // Test: Numerical results match CPU reference
}
```

### Integration Tests (End-to-End)

```rust
#[test]
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn test_gpu_attention_forward() {
    // 1. Initialize GPU backend
    let mut kernels = UnifiedGpuKernels::auto_detect()?;
    
    // 2. Prepare test data
    let input = random_tensor((batch, seq, embed));
    let weights = random_tensor(...);
    
    // 3. Execute GPU forward
    let gpu_output = kernels.attention_forward(
        &input, &wq, &wk, &wv, &wo, &params
    )?;
    
    // 4. Compare with CPU reference
    let cpu_output = cpu_attention_forward(...);
    assert_close!(gpu_output, cpu_output, eps=1e-4);
}
```

### Benchmark Tests

```rust
#[bench]
fn bench_richards_curve_wgpu(b: &mut Bencher) {
    let mut device = GpuDevice::auto_detect().unwrap();
    let input = device.allocate_f32(1024 * 1024).unwrap();
    let mut output = device.allocate_f32(1024 * 1024).unwrap();
    
    b.iter(|| {
        device.richards_curve(&input, &mut output, &params, 1024 * 1024)
    });
}

// Expected results:
// - 10GB+/s memory bandwidth utilization
// - < 1ms execution time for 1M elements
// - 25x speedup vs CPU (50ms → 2ms)
```

---

## 9. Troubleshooting with No-Fallback Mode

### Debugging Workflow

```
Problem: "GPU operation failed"
├─ Is GPU available?
│  └─ `GpuDevice::auto_detect()` fails
│     → Install graphics drivers or run with GPU
│
├─ Is feature flag enabled?
│  └─ `cargo build --features wgpu`
│     → Check Cargo.toml, enable matching backend
│
├─ Does kernel compile?
│  └─ WGSL shader compile error
│     → Check WGSL syntax, validate shaders in wgsl_kernels.rs
│
└─ Is output correct?
   └─ GPU result ≠ CPU reference
      → Numerical precision issue
      → Check stable exponential handling
      → Validate parameter passing
```

### Common Error Messages

```
# No GPU detected
Error: Automatic GPU detection failed: no supported GPU backend detected (CUDA/Metal/Vulkan).

→ Solution: Install GPU drivers, or run on GPU-enabled system

# Missing feature flag
Error: CUDA backend requires cudarc feature. Compile with --features gpu-cuda

→ Solution: cargo build --release --features gpu-cuda

# Shader compilation failure
Error: WGPU shader compilation failed for 'SHADER_RICHARDS_CURVE': 
       error at offset X: unknown identifier 'richards_activation'

→ Solution: Check wgsl_kernels.rs for undefined functions

# Dimension mismatch
Error: GPU kernel execution error: invalid dimensions for kernel dispatch.
       Expected: batch=64, embed=1024. Got: batch=33, embed=769.

→ Solution: Check workspace capacity, ensure_capacity may need explicit resize
```

---

## 10. Implementation Checklist

### Phase 3A: Fused Kernels

- [ ] **RichardsGLU Fused Pass 1**
  - [ ] WGSL shader implementation
  - [ ] Parameter struct (OptimizedRichardsGluParams)
  - [ ] CPU reference implementation for validation
  - [ ] Unit tests

- [ ] **RichardsGLU Fused Pass 2**
  - [ ] GEMM + output projection
  - [ ] Zero-copy data handling
  - [ ] Integration with Pass 1

- [ ] **CUDA Kernel Stub**
  - [ ] File: `src/domain/compute/kernels/richards_glu.cu`
  - [ ] Template implementation
  - [ ] Compilation integration

- [ ] **Metal Kernel Stub**
  - [ ] File: `src/domain/compute/kernels/richards_glu.metal`
  - [ ] Template implementation
  - [ ] Compilation integration

### Phase 3B: Workspace Management

- [ ] **GpuKernelWorkspace**
  - [ ] Power-of-2 sizing
  - [ ] Buffer lifecycle management
  - [ ] Capacity tracking
  - [ ] Statistics collection

- [ ] **UnifiedGpuKernels Integration**
  - [ ] `ensure_capacity()`
  - [ ] `reset_workspace()`
  - [ ] `cleanup_workspace()`
  - [ ] `workspace_stats()`

### Phase 3C: GPU Detection

- [ ] **Strict No-Fallback**
  - [ ] All GPU ops return explicit errors
  - [ ] No silent CPU fallback
  - [ ] Informative error messages

- [ ] **Feature Flag Handling**
  - [ ] Compile-time detection
  - [ ] Runtime detection
  - [ ] Mismatch error reporting

### Phase 3D: Testing

- [ ] **Unit Tests**
  - [ ] Kernel correctness
  - [ ] Workspace management
  - [ ] Error handling

- [ ] **Integration Tests**
  - [ ] End-to-end forward pass
  - [ ] GPU vs CPU accuracy
  - [ ] Multiple backends

- [ ] **Performance Benchmarks**
  - [ ] Kernel execution time
  - [ ] Memory bandwidth
  - [ ] Speedup vs CPU

---

## 11. Code Examples

### Using UnifiedGpuKernels

```rust
use crate::domain::layers::components::unified_gpu_kernels::{
    UnifiedGpuKernels, AttentionParams, SsmParams,
};

// Create with automatic GPU detection
let mut kernels = UnifiedGpuKernels::auto_detect()?;

// Prepare workspace for computation
kernels.ensure_capacity(batch_size, embed_dim, seq_len)?;

// Execute attention
let output = kernels.attention_forward(
    &input,
    &wq, &wk, &wv, &wo,
    &AttentionParams::new(num_heads, embed_dim, seq_len, batch_size),
)?;

// Cleanup when done
kernels.cleanup_workspace()?;
```

### Implementing a New GPU Kernel

```rust
// 1. Define WGSL shader in wgsl_kernels.rs
pub const SHADER_MY_KERNEL: &str = r#"
    // WGSL shader code
"#;

// 2. Add trait method to GpuMatrixOps
pub trait GpuMatrixOps {
    fn my_operation(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &MyParams,
        size: usize,
    ) -> Result<()>;
}

// 3. Implement for WGPU backend in wgpu_ops.rs
impl GpuMatrixOps for WgpuMatrixOps {
    fn my_operation(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &MyParams,
        size: usize,
    ) -> Result<()> {
        // Create bind group, pipeline, dispatch workgroups
        // Wait for completion
        Ok(())
    }
}

// 4. Add CPU stubs for CUDA/Metal (fill in later)
#[cfg(feature = "gpu-cuda")]
impl GpuMatrixOps for CudaMatrixOps {
    fn my_operation(...) -> Result<()> {
        Err(ModelError::Backend {
            message: "my_operation not yet implemented for CUDA".to_string(),
        })
    }
}
```

---

## 12. Performance Targets & Validation

### Expected Speedups

| Operation | CPU Time | GPU Target | Speedup |
|---|---|---|---|
| RichardsGLU (1K batch) | 50ms | 2ms | 25x |
| PolyAttention (512 batch) | 30ms | 1ms | 30x |
| Mamba Selective Scan (512 batch) | 40ms | 2ms | 20x |
| RG-LRU Recurrent (512 batch) | 30ms | 2ms | 15x |

### Validation Criteria

- ✅ GPU output matches CPU reference (ε ≤ 1e-4)
- ✅ Measured speedup ≥ 10x (conservative target)
- ✅ Memory bandwidth utilization ≥ 60%
- ✅ No silent fallbacks to CPU
- ✅ All backends (WGPU, CUDA, Metal) pass tests

