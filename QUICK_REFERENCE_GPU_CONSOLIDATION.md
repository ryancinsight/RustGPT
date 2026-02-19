# GPU Consolidation Quick Reference - Phase 5.6.3

## 🎯 Quick Navigation

| Need | File | Link |
|------|------|------|
| **How to create a GPU kernel?** | `GPU_KERNEL_CONSOLIDATION_IMPLEMENTATION_GUIDE.md` | 5-step pattern |
| **What's the roadmap?** | `CONSOLIDATION_GPU_KERNELS_PHASE5.6.3_EXECUTION.md` | Phases 3A-3D |
| **Session status?** | `SESSION_GPU_CONSOLIDATION_STATUS_FEB16.md` | Completed work |
| **Summary?** | `GPU_CONSOLIDATION_SUMMARY_FEB16.md` | Overview |

## 🚀 Quick Start (Copy-Paste Ready)

### Enable GPU Auto-Detection
```rust
let kernels = UnifiedGpuKernels::auto_detect()?;
```

### Create Workspace
```rust
kernels.ensure_capacity(batch_size, embed_dim, seq_len)?;
```

### Use GPU Operations
```rust
let output = kernels.activation_forward(&input, GpuActivation::Gelu)?;
```

### Get Memory Stats
```rust
let stats = kernels.workspace_stats();
println!("Memory: {:.1} MB", stats.estimated_memory_bytes as f64 / 1024.0 / 1024.0);
```

### Cleanup
```rust
kernels.cleanup_workspace()?;
```

## 📋 GPU Backend Priority Order
```
CUDA (Preferred for NVIDIA GPUs)
  ↓ if not available
Metal (Preferred for Apple Silicon/macOS)
  ↓ if not available
Vulkan/WGPU (Fallback for cross-platform)
  ↓ if not available
ERROR (No GPU - strict no-fallback)
```

## 🔧 Create New GPU Kernel (5 Steps)

### Step 1: Define Parameters
```rust
#[derive(Debug, Clone)]
pub struct MyKernelParams {
    pub batch_size: usize,
    pub embed_dim: usize,
    pub seq_len: usize,
    pub temperature: f32,
}
```

### Step 2: CPU Reference Implementation
```rust
pub fn forward_reference_cpu(
    input: &Array2<f32>,
    params: &MyKernelParams,
) -> Result<Array2<f32>> {
    // CPU version for validation
    Ok(output)
}
```

### Step 3: GPU Implementation
```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn forward_gpu(
    device: &mut GpuDevice,
    input: &GpuBuffer,
    params: &MyKernelParams,
) -> Result<GpuBuffer> {
    let mut output = device.allocate(output_size)?;
    device.my_kernel_op(input, &mut output, params)?;
    Ok(output)
}
```

### Step 4: Integration into UnifiedGpuKernels
```rust
impl UnifiedGpuKernels {
    pub fn my_kernel_forward(
        &mut self,
        input: &Array2<f32>,
        params: &MyKernelParams,
    ) -> Result<Array2<f32>> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock".to_string(),
        })?;
        
        self.ensure_capacity(params.batch_size, params.embed_dim, params.seq_len)?;
        
        let input_size = params.batch_size * params.embed_dim * std::mem::size_of::<f32>();
        let mut input_buf = device.allocate(input_size)?;
        device.upload(input.as_slice().unwrap(), &mut input_buf)?;
        
        let output_buf = forward_gpu(&mut device, &input_buf, params)?;
        
        let mut output = vec![0.0f32; params.batch_size * params.embed_dim];
        device.download(&output_buf, &mut output)?;
        
        device.deallocate(input_buf);
        device.deallocate(output_buf);
        
        Ok(Array2::from_shape_vec((params.batch_size, params.embed_dim), output)?)
    }
}
```

### Step 5: Add Tests
```rust
#[test]
fn test_my_kernel_gpu() {
    if UnifiedGpuKernels::auto_detect().is_err() {
        println!("No GPU, skipping test");
        return;
    }
    
    let mut kernels = UnifiedGpuKernels::auto_detect().unwrap();
    let input = Array2::<f32>::ones((8, 64));
    let params = MyKernelParams { /* ... */ };
    
    kernels.ensure_capacity(8, 64, 32).unwrap();
    let gpu_output = kernels.my_kernel_forward(&input, &params).unwrap();
    let cpu_output = forward_reference_cpu(&input, &params).unwrap();
    
    // Verify match
    let max_diff = gpu_output.iter().zip(cpu_output.iter())
        .map(|(g, c)| (g - c).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    assert!(max_diff < 1e-4);
    
    kernels.cleanup_workspace().unwrap();
}
```

## 🎯 Implementation Priority Matrix

### Priority 1 (Do First - Highest Impact)
| Kernel | Target Speedup | Complexity | File |
|--------|---|---|---|
| Attention | 30x | HIGH | `attention_gpu_kernel.rs` |
| RichardsGLU | 25x | MEDIUM | Enhance `richards_glu_fused_kernel.rs` |

### Priority 2 (Do Next - High Impact)
| Kernel | Target Speedup | Complexity | File |
|--------|---|---|---|
| Selective Scan | 20x | VERY HIGH | `mamba_selective_scan_gpu.rs` |
| RG-LRU | 15x | HIGH | `rg_lru_gpu_kernel.rs` |

### Priority 3 (Foundation - Low Impact)
| Kernel | Target Speedup | Complexity | File |
|--------|---|---|---|
| Normalization | 5-10x | LOW | `norm_gpu_kernel.rs` |

## 📊 Memory Calculation Quick Reference

**Standard 8-buffer workspace:**
```
activation_0:  batch * embed * 4 bytes
activation_1:  batch * embed * 4 bytes
qkv_0:         batch * embed * 4 bytes
qkv_1:         batch * embed * 4 bytes
qkv_2:         batch * embed * 4 bytes
scores:        batch * seq * seq * 4 bytes  ← Usually largest
attn_output:   batch * embed * 4 bytes
weight:        embed * embed * 4 bytes
```

**Example: batch=512, embed=768, seq=512**
```
activation: 2 * 512 * 768 * 4 = 3.1 MB
qkv:        3 * 512 * 768 * 4 = 4.7 MB
scores:     512 * 512 * 512 * 4 = 512 MB  ← Watch this!
attn_out:   512 * 768 * 4 = 1.5 MB
weight:     768 * 768 * 4 = 2.3 MB
TOTAL:                          ~523 MB
```

## ✅ Build & Test Commands

```bash
# Check compilation (0 errors, 0 warnings target)
cargo check --lib

# Run all tests
cargo test --lib

# Run single test
cargo test --lib test_my_kernel_gpu -- --exact --nocapture

# Build with GPU backends
cargo build --release --features gpu-all  # All backends
cargo build --release --features gpu-cuda # NVIDIA only
cargo build --release --features gpu-wgpu # Cross-platform
```

## 🔍 Debugging Tips

### Check GPU Availability
```rust
match GpuDevice::auto_detect() {
    Ok(device) => println!("GPU: {}", device.backend().as_str()),
    Err(e) => println!("No GPU: {}", e),
}
```

### Monitor Workspace Memory
```rust
let stats = kernels.workspace_stats();
println!("Buffers: {}", stats.buffer_count);
println!("Memory: {:.1} MB", stats.estimated_memory_bytes as f64 / 1024.0 / 1024.0);
println!("Allocations: {} total, {} reallocations", 
    stats.allocation_count, stats.reallocation_count);
```

### Validate GPU vs CPU
```rust
let gpu_output = kernels.my_kernel_forward(&input, &params)?;
let cpu_output = forward_reference_cpu(&input, &params)?;

let max_diff = gpu_output.iter().zip(cpu_output.iter())
    .map(|(g, c)| (g - c).abs())
    .max_by(|a, b| a.partial_cmp(b).unwrap())
    .unwrap();

println!("Max difference: {:.2e}", max_diff);
assert!(max_diff < 1e-4, "Outputs differ too much!");
```

## 🚨 Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| GPU not detected | Feature flags not enabled | `cargo build --features gpu-cuda` |
| Memory exhaustion | Buffers too large | Use power-of-2 sizing |
| Sync errors | Missing lock acquisition | Check `.lock().map_err()` pattern |
| Numerical differences | Float precision | Use tolerance in tests (~1e-4) |
| Kernel not available | GPU backend missing | Check `#[cfg(...)]` guards |

## 📚 Reference Files

**Main implementation files:**
- `src/domain/layers/components/unified_gpu_kernels.rs` - Main dispatcher
- `src/domain/layers/components/unified_gpu_backend.rs` - Backend traits
- `src/domain/compute/gpu_device.rs` - GPU device context

**Reference implementations:**
- `src/domain/compute/richards_glu_fused_kernel.rs` - Two-pass fused kernel pattern
- `src/domain/layers/components/gpu_shared_executor.rs` - Workspace management

**Test examples:**
- `src/domain/layers/components/unified_gpu_kernels.rs` (tests module)
- `src/domain/compute/richards_glu_fused_kernel.rs` (tests module)

## 🎓 Learning Path

1. **Understand**: Read `GPU_KERNEL_CONSOLIDATION_IMPLEMENTATION_GUIDE.md`
2. **Review**: Look at `richards_glu_fused_kernel.rs` (working example)
3. **Create**: Follow 5-step pattern for new kernel
4. **Test**: Use provided testing template
5. **Profile**: Measure speedup vs CPU
6. **Optimize**: Adjust parameters, fuse operations

## ✨ Success Criteria

Each GPU kernel should achieve:
- ✅ Compiles without warnings
- ✅ 0 errors on `cargo check --lib`
- ✅ Matches CPU reference output (tolerance < 1e-4)
- ✅ Achieves target speedup (typically 15-30x)
- ✅ Passes all tests: `cargo test --lib`
- ✅ No CPU fallback (GPU-only)

---

**Last Updated**: Feb 16, 2026  
**Thread**: @T-019c6753-5d92-72de-b050-d422c54bfd65  
**Status**: Ready for Phase 3B implementation

