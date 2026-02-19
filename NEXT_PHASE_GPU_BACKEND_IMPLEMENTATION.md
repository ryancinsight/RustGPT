# Next Phase: GPU Backend Implementation (Phase 5.6+)

**Current Status**: ✅ Compilation successful, zero errors  
**Ready For**: GPU-specific backend implementations

---

## Immediate Next Steps (This Session)

### 1. Suppress Deprecation Warnings (5 min)
Add `#[allow(deprecated)]` to test files using `CpuGpuMatrixOps`:
- `src/domain/layers/components/attention_context_gpu.rs:159, 175`
- `src/domain/layers/components/feedforward_gpu.rs:124`

### 2. Clean Up Minor Warnings (10 min)
Remove unused imports:
```rust
// unified_gpu_buffer_pool.rs:420 - Remove
use super::*;

// unified_gpu_executor.rs:386 - Remove  
use super::*;
```

### 3. Run Integration Tests (5 min)
```bash
cargo test --lib
```
Verify all tests pass with warnings-only output.

---

## GPU Backend Implementation Roadmap

### Phase 5.6: WGPU Backend Foundation (Est. 8-12 hours)
**Goal**: Cross-platform GPU support via WebGPU

**Tasks**:
1. Implement `GpuDevice::auto_detect()` for WGPU
2. Create WGPU `GpuMatrixOps` implementation struct
3. Implement BLAS Level 3 operations (gemm_f32, gemm_batched_f32, gemv_f32)
4. Add shader compilation pipeline

**Key Files**:
- Create: `src/domain/compute/wgpu_backend.rs`
- Create: `src/domain/compute/wgpu_kernels/` (shader collection)
- Modify: `src/domain/compute/gpu_device.rs` (auto-detect logic)

**Testing**: Unit tests for each operation with CPU reference validation

---

### Phase 5.7: CUDA Backend (Est. 10-14 hours)
**Goal**: NVIDIA GPU acceleration via cuBLAS + custom kernels

**Tasks**:
1. Add `cust` crate bindings for CUDA
2. Implement CUDA `GpuMatrixOps` using cuBLAS
3. Create custom CUDA kernels for Richards curve, PolyAttention ops
4. Add device capability checking

**Key Files**:
- Create: `src/domain/compute/cuda_backend.rs`
- Create: `src/domain/compute/cuda_kernels/` (PTX/CUBIN)
- Modify: GPU device auto-detection

**Dependencies**:
```toml
cust = "0.2"  # CUDA bindings
```

---

### Phase 5.8: Metal Backend (Est. 8-10 hours)
**Goal**: Apple Silicon support

**Tasks**:
1. Implement Metal Performance Shaders `GpuMatrixOps`
2. Create Metal shaders for all operations
3. Thread safety via Metal command queues
4. Unified memory management

**Key Files**:
- Create: `src/domain/compute/metal_backend.rs`
- Create: `src/domain/compute/metal_shaders/` (Metal shader library)

**Dependencies**:
```toml
metal = "0.27"  # macOS only
```

---

## Critical Implementation Patterns

### No-Fallback Error Pattern
```rust
impl GpuMatrixOps for WgpuOps {
    fn gemm_f32(...) -> Result<()> {
        // Check device is ready
        let device = self.device.as_ref()
            .ok_or(ModelError::Backend {
                message: "GPU device not initialized".into()
            })?;
        
        // Compute or return error, never CPU fallback
        self.compute_gemm(device, ...)
    }
}
```

### Strict GPU Device Initialization
```rust
// GOOD: Explicit GPU requirement
let gpu_device = GpuDevice::auto_detect()?;
let ops = gpu_device.create_ops()?;
let result = ops.gemm_f32(...)?;

// BAD: Implicit fallback (don't do this)
let result = if has_gpu {
    ops.gemm_f32(...)
} else {
    cpu_gemm(...)  // ❌ FORBIDDEN
};
```

### Numerical Accuracy Requirements
```rust
// All GPU operations must match CPU reference within ε ≤ 1e-4
#[cfg(test)]
fn validate_gpu_vs_cpu(gpu_result: &[f32], cpu_result: &[f32]) {
    for (g, c) in gpu_result.iter().zip(cpu_result) {
        assert!((g - c).abs() <= 1e-4, 
            "GPU result {g} differs from CPU {c} beyond threshold");
    }
}
```

---

## Memory Management Requirements

### Buffer Pool Allocation
```
┌─ UnifiedGpuBufferPool
│  ├─ Power-of-2 sizing for efficient reuse
│  ├─ LRU eviction for unused buffers
│  ├─ Thread-local caching (no synchronization overhead)
│  └─ Batch size tracking (minimize reallocations)
└─ Target: <1% of compute time on transfers
```

### Streaming Workspace Integration
```rust
// Pre-allocate for streaming inference
let mut workspace = PolyAttentionStreamingWorkspace::new();
for token in stream {
    let output = attention.forward_streaming_gpu(token, &mut workspace)?;
}
```

---

## Validation Benchmarks

### Performance Targets
| Operation | Target TFLOPS | CPU Baseline | GPU Speedup |
|-----------|---------------|--------------|-------------|
| gemm_f32  | 50-100+       | 2-5 TFLOPS   | 10-50x      |
| layer_norm| 100-200 GB/s  | Single core  | 10-30x      |
| softmax   | 50-100 GB/s   | Single core  | 5-20x       |

### Memory Efficiency
- Peak GPU memory: <90% of capacity
- Buffer reuse rate: >80%
- Allocation frequency: <0.1% of forward passes

---

## Testing Strategy

### Unit Tests (Per Operation)
```rust
#[test]
fn test_wgpu_gemm_f32_correctness() {
    // Create CPU reference result
    let cpu_result = gemm_cpu(...);
    
    // Create GPU result
    let device = GpuDevice::auto_detect()?;
    let ops = device.create_ops()?;
    let gpu_result = ops.gemm_f32(...)?;
    
    // Validate
    assert_gpu_vs_cpu(&gpu_result, &cpu_result, 1e-4);
}
```

### Integration Tests
```rust
#[test]
fn test_poly_attention_forward_gpu() {
    let mut attention = PolyAttention::new(...);
    attention.ensure_gpu_device_auto_detect()?;
    
    let gpu_result = attention.forward_gpu(...)?;
    let cpu_result = attention.forward(...);
    
    validate_outputs(&gpu_result, &cpu_result);
}
```

### Benchmark Suite
```bash
cargo bench --bench gpu_operations
```

---

## Branch Strategy

```
main (stable, CPU-only)
├─ phase-5.6-wgpu (WGPU implementation)
├─ phase-5.7-cuda (CUDA implementation)  
└─ phase-5.8-metal (Metal implementation)
```

Merge phase branches into `main` once validated.

---

## Dependency Management

### Feature Flags
```toml
[features]
default = ["cpu"]
cpu = []
gpu = ["gpu-wgpu"]
gpu-wgpu = ["wgpu", "wgsl-in"]
gpu-cuda = ["cust", "cuda-sys"]
gpu-metal = ["metal"]
```

Enable via:
```bash
cargo build --features gpu-wgpu
cargo build --features gpu-cuda
cargo build --features gpu-metal
```

---

## Documentation Requirements

1. **GPU Architecture Diagram** - Block diagram of device, buffer pool, executor
2. **Shader Documentation** - Comments for all WGSL/GLSL/Metal kernels
3. **API Migration Guide** - How to move from CPU to GPU code paths
4. **Troubleshooting Guide** - Common GPU errors and solutions
5. **Performance Tuning** - Block size, workgroup sizing, memory bandwidth

---

## Risk Mitigations

### Risk: Shader Compilation Failures
- **Mitigation**: Pre-compile shaders at build time, cache compiled binaries
- **Test**: `cargo test --lib gpu_shader_compilation`

### Risk: GPU Memory Exhaustion
- **Mitigation**: LRU eviction + configurable max capacity
- **Monitor**: Log peak memory usage per forward pass

### Risk: Numerical Instability
- **Mitigation**: Strict ε ≤ 1e-4 validation, reference tests on each operation
- **Fallback**: Fallback to CPU for individual operations (local fallback only, not full-path)

### Risk: Cross-Platform Incompatibility
- **Mitigation**: Separate backend implementations with unified interface
- **Test**: CI/CD runs on Linux, macOS, Windows

---

## Session Handoff Checklist

- [x] Compilation successful (0 errors)
- [x] GPU trait definitions complete
- [x] Deprecation stubs in place
- [ ] Suppress remaining warnings
- [ ] Run full test suite
- [ ] Document GPU backend structure
- [ ] Create WGPU skeleton
- [ ] Set up shader compilation pipeline

**Ready for Phase 5.6 (WGPU backend) implementation.**
