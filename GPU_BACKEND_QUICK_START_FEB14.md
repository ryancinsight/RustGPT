# GPU Backend Quick Start - Ready for Phase 5.6

## Status: ✅ READY FOR GPU IMPLEMENTATION

---

## What Was Done

All 15 compilation errors have been fixed. The codebase now:
- ✅ Compiles cleanly (0 errors)
- ✅ Has unified GPU trait system
- ✅ Exports all shared components
- ✅ Validates type safety
- ✅ Enforces no-fallback GPU policy

---

## Quick Wins First (10 minutes)

### 1. Suppress Deprecation Warnings
Add to these test files:
```rust
// File: src/domain/layers/components/attention_context_gpu.rs:150
#[cfg(test)]
mod tests {
    use super::*;
    #[allow(deprecated)]  // CpuGpuMatrixOps is deprecated stub
    ...
}

// File: src/domain/layers/components/feedforward_gpu.rs:100
#[cfg(test)]
mod tests {
    use super::*;
    #[allow(deprecated)]  // CpuGpuMatrixOps is deprecated stub
    ...
}
```

### 2. Clean Up Unused Imports
```rust
// File: src/domain/compute/unified_gpu_buffer_pool.rs:420
// REMOVE: use super::*;

// File: src/domain/compute/unified_gpu_executor.rs:386
// REMOVE: use super::*;
```

Then run:
```bash
cargo check
```

---

## GPU Trait System (Your Starting Point)

### The Core Trait
```rust
// File: src/domain/compute/gpu_ops.rs
pub trait GpuMatrixOps: Send + Sync {
    // BLAS Level 3: Matrix-Matrix
    fn gemm_f32(...) -> Result<()>;
    fn gemm_batched_f32(...) -> Result<()>;
    fn gemv_f32(...) -> Result<()>;
    
    // Element-wise
    fn relu(...) -> Result<()>;
    fn gelu(...) -> Result<()>;
    fn silu(...) -> Result<()>;
    fn sigmoid(...) -> Result<()>;
    fn mul(...) -> Result<()>;
    fn add_scaled(...) -> Result<()>;
    fn scale(...) -> Result<()>;
    fn axpy(...) -> Result<()>;
    fn richards_curve(...) -> Result<()>;
    
    // Normalization
    fn layer_norm(...) -> Result<()>;
    fn softmax(...) -> Result<()>;
    
    // Reductions
    fn sum(...) -> Result<f32>;
    fn mean(...) -> Result<f32>;
    
    // Transfers
    fn download(...) -> Result<()>;
    fn upload(...) -> Result<()>;
    fn copy_within_device(...) -> Result<()>;
    
    // Permutation
    fn permute_4d(...) -> Result<()>;
    
    // PolyAttention-specific
    fn poly_attention_fused(...) -> Result<()>;
    fn blr_projection(...) -> Result<()>;
    fn compute_cope_scores(...) -> Result<()>;
    fn moh_gate_activation(...) -> Result<()>;
}
```

### CPU Fallback for Testing
```rust
pub struct CpuMatrixOps;  // Located at ~line 400 in gpu_ops.rs

impl GpuMatrixOps for CpuMatrixOps {
    // Returns errors on all operations
    // Use for testing GPU-required code paths only
}
```

### GPU Device Interface (To Implement)
```rust
pub struct GpuDevice {
    // WGPU: device, queue, adapter
    // CUDA: device handle
    // Metal: device, command queue
}

impl GpuDevice {
    pub fn auto_detect() -> Result<Self> {
        // TODO: Implement in Phase 5.6
        // 1. Check for WGPU support
        // 2. Check for CUDA support  
        // 3. Check for Metal support
        // Return first available, error if none
    }
    
    pub fn create_ops(&self) -> Result<Box<dyn GpuMatrixOps>> {
        // TODO: Return backend-specific implementation
    }
}
```

---

## Component Integration Points

### Where Shared Components Attach GPU

#### 1. Poly Attention (attention_context_gpu.rs)
```rust
pub struct SharedAttentionContext {
    // CPU fields (exists)
}

impl SharedAttentionContext {
    pub fn apply_incoming_context_gpu(
        &self, 
        input: &Array2<f32>,
        ctx: &mut GpuSharedOpsContext,
        ops: &mut dyn GpuMatrixOps,  // You fill this in
    ) -> Result<Array2<f32>> {
        // TODO: Use ops.gemm_f32, ops.layer_norm, etc.
    }
}
```

#### 2. Richards Feedforward (feedforward_gpu.rs)
```rust
pub enum FeedForwardVariant {
    RichardsGlu(Box<RichardsGlu>),  // Now works
    MixtureOfExperts(Box<MixtureOfExperts>),
}

impl RichardsGlu {
    pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // TODO: Implement GPU path using:
        // - ops.gemm_f32() for matrix ops
        // - ops.richards_curve() for activation
        // - ops.layer_norm() for normalization
    }
}
```

#### 3. Temporal Processing (temporal_processing_gpu.rs)
```rust
impl PolyAttention {
    pub fn forward_gpu_with_ops(
        &mut self,
        input: &Array2<f32>,
        ops: &mut dyn GpuMatrixOps,  // You fill this in
    ) -> Result<Array2<f32>> {
        // TODO: Full PolyAttention GPU forward pass
        // - Use ops.poly_attention_fused() for main computation
        // - Use ops.moh_gate_activation() for gating
        // - Use ops.permute_4d() for head arrangement
    }
}
```

---

## No-Fallback Policy (CRITICAL)

### ✅ DO THIS
```rust
// Explicit GPU requirement
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // Get GPU device (will error if not available)
    let device = GpuDevice::auto_detect()?;
    let mut ops = device.create_ops()?;
    
    // Use GPU ops
    ops.gemm_f32(self.pool, 1.0, &w, &input, 0.0, &mut output, ...)?;
    
    Ok(output)  // Or error, never CPU fallback
}
```

### ❌ DON'T DO THIS
```rust
// Silent fallback (violates GPU policy)
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    match GpuDevice::auto_detect() {
        Ok(device) => {
            let mut ops = device.create_ops()?;
            ops.gemm_f32(...)  // GPU
        }
        Err(_) => {
            // ❌ FORBIDDEN: CPU fallback
            self.forward(input)  // Should return Err instead!
        }
    }
}
```

---

## Implementation Priority Order

### Must-Have (Phase 5.6)
1. **GpuDevice auto-detection**
   - File: `src/domain/compute/gpu_device.rs`
   - WGPU feature detection
   - CUDA capability checking
   - Metal availability

2. **WGPU Backend**
   - File: Create `src/domain/compute/wgpu_backend.rs`
   - Implement all BLAS operations
   - Shader compilation
   - Buffer management

3. **Numerical Validation**
   - CPU vs GPU comparison
   - ε ≤ 1e-4 tolerance
   - Test suite for each operation

### Should-Have (Phase 5.7+)
4. **CUDA Backend** - nvidia/cuBLAS
5. **Metal Backend** - Apple Silicon
6. **Performance Tuning** - Block sizes, memory bandwidth

### Nice-to-Have
7. **Kernel Fusion** - GEMM + Activation
8. **Async Execution** - Pipelined operations
9. **Profiling** - Operation-level timing

---

## File Map

### Core GPU Files
```
src/domain/compute/
├── gpu_ops.rs           ← Trait definitions (READY ✅)
├── gpu_device.rs        ← Device detection (TODO)
├── gpu_memory.rs        ← Buffer types (exists)
├── unified_gpu_buffer_pool.rs  ← Memory mgmt (exists)
├── unified_gpu_executor.rs     ← Executor (exists)
├── wgpu_backend.rs      ← WGPU impl (TODO)
├── cuda_backend.rs      ← CUDA impl (TODO)
└── metal_backend.rs     ← Metal impl (TODO)
```

### GPU-Accelerated Layers
```
src/domain/layers/
├── components/
│   ├── attention_context_gpu.rs  ← SharedAttentionContext (READY ✅)
│   ├── feedforward_gpu.rs        ← RichardsGlu GPU (READY ✅)
│   └── temporal_processing_gpu.rs ← PolyAttention GPU (READY ✅)
└── ...
```

---

## Testing During Development

### Unit Test Template
```rust
#[test]
fn test_gpu_gemm_correctness() {
    // Setup
    let device = GpuDevice::auto_detect().expect("GPU required");
    let mut ops = device.create_ops().expect("Failed to create ops");
    
    // GPU computation
    let gpu_result = ops.gemm_f32(...)?;
    
    // CPU reference
    let cpu_result = gemm_cpu(...);
    
    // Validate within tolerance
    assert_gpu_vs_cpu(&gpu_result, &cpu_result, 1e-4);
}
```

### Run Tests
```bash
cargo test --lib gpu_

# With output
cargo test --lib gpu_ -- --nocapture

# Specific operation
cargo test --lib test_gpu_gemm_correctness
```

---

## Debugging Tips

### If GPU Not Detected
```rust
// Add debug output in auto_detect()
eprintln!("Checking WGPU availability...");
if !wgpu::is_supported() {
    eprintln!("WGPU not available on this system");
    return Err(ModelError::Backend { ... });
}
```

### If Numerical Mismatch
```rust
// Compare element-wise
for (i, (gpu, cpu)) in gpu_result.iter().zip(&cpu_result).enumerate() {
    let diff = (gpu - cpu).abs();
    if diff > 1e-4 {
        eprintln!("Mismatch at {}: GPU={}, CPU={}, diff={}", i, gpu, cpu, diff);
    }
}
```

### If Shader Compilation Fails
```rust
// Add shader validation before use
let shader = std::fs::read_to_string("shader.wgsl")?;
validate_wgsl(&shader)?;  // Add validation function
```

---

## Success Criteria for Phase 5.6

- [x] Compilation passes with 0 errors
- [ ] GPU device auto-detection working
- [ ] WGPU backend compiles
- [ ] All BLAS ops tested vs CPU reference
- [ ] Numerical accuracy ε ≤ 1e-4
- [ ] SharedAttentionContext uses GPU ops
- [ ] RichardsGlu forward_gpu implemented
- [ ] PolyAttention forward_gpu implemented

---

**You're ready to build the GPU backend. Start with GpuDevice::auto_detect().**
