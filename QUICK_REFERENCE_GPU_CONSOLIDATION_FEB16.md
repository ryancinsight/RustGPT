# GPU Consolidation - Quick Reference Guide
**Phase 5.6** | **Feb 16, 2026** | **No-Fallback GPU Semantics**

## 📍 Key Files & Locations

### Core GPU Infrastructure
| File | Purpose | Status |
|------|---------|--------|
| `src/domain/compute/gpu_device.rs` | GpuDevice + auto-detect | ⚠️ Needs consolidation |
| `src/domain/compute/gpu_ops.rs` | GpuMatrixOps trait | ✓ Complete |
| `src/domain/compute/unified_gpu_buffer_pool.rs` | Memory pooling | ⚠️ Needs unification |
| `src/domain/compute/gpu_component.rs` | GpuComponent trait | ✓ Complete |

### Shared Components (GPU Variants)
| File | Purpose | Status |
|------|---------|--------|
| `src/domain/layers/components/attention_context.rs` | CPU impl | ✓ Complete |
| `src/domain/layers/components/attention_context_gpu.rs` | GPU impl | 🔴 70% |
| `src/domain/layers/components/feedforward.rs` | CPU impl | ✓ Complete |
| `src/domain/layers/components/feedforward_gpu.rs` | GPU impl | 🔴 40% |
| `src/domain/layers/components/temporal_processing.rs` | CPU impl | ✓ Complete |
| `src/domain/layers/components/temporal_processing_gpu.rs` | GPU impl | 🔴 20% |

### Unified Dispatchers
| File | Purpose | Status |
|------|---------|--------|
| `src/domain/layers/components/unified_gpu_kernels.rs` | Kernel dispatcher | ✓ Complete |
| `src/domain/layers/components/unified_gpu_backend.rs` | Backend dispatcher | ✓ Complete |

---

## 🔧 Implementation Patterns

### Pattern A: GPU Forward with Workspace
```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let mut device = self.device().lock().map_err(|_| ModelError::Backend {
        message: "GPU device lock failed in ComponentName::forward_gpu".to_string(),
    })?;

    let (batch_size, embed_dim) = input.dim();
    
    // Ensure workspace has capacity (power-of-2 sizing, reuses buffers)
    let workspace = self.workspace.as_mut().ok_or(ModelError::Backend {
        message: "GPU workspace not initialized".to_string(),
    })?;
    workspace.ensure_capacity(&mut device, batch_size, embed_dim, seq_len)?;

    // Upload input
    device.upload(input.as_slice().unwrap(), &mut workspace.buf_input)?;

    // Execute kernel operation (reuses workspace buffers)
    device.gemm_f32(
        1.0, &workspace.buf_input, &workspace.w1,
        0.0, &mut workspace.buf_hidden,
        batch_size, hidden_dim, embed_dim, false, false
    )?;

    // Activation on GPU (not CPU post-download)
    device.relu(&workspace.buf_hidden, &mut workspace.buf_hidden, batch_size * hidden_dim)?;

    // Download result
    let mut output = vec![0.0f32; batch_size * embed_dim];
    device.download(&workspace.buf_output, &mut output)?;

    // Update telemetry
    self.stats.kernel_launches += 2;
    self.stats.bytes_uploaded += input.len() * std::mem::size_of::<f32>();
    self.stats.bytes_downloaded += output.len() * std::mem::size_of::<f32>();

    Ok(Array2::from_shape_vec((batch_size, embed_dim), output)?)
}
```

### Pattern B: Main Forward with No-Fallback GPU
```rust
pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // GPU path is primary (no fallback to CPU)
    if self.gpu_device.is_some() {
        return self.forward_gpu(input)
            .expect("GPU forward failed - no CPU fallback for strict GPU mode");
    }
    
    // CPU path when GPU not available
    self.forward_cpu(input)
}
```

### Pattern C: Auto-Detect with Feature Flags
```rust
impl MyComponent {
    pub fn auto_detect() -> Result<Self> {
        // Initialize GPU backend (errors if no GPU available)
        let device = GpuDevice::auto_detect()?;
        
        let mut component = Self::new();
        component.gpu_device = Some(Arc::new(Mutex::new(device)));
        component.workspace = Some(ComponentWorkspace::new());
        
        Ok(component)
    }
}

// Usage:
let mut component = MyComponent::auto_detect()?;  // Errors if no GPU
```

---

## ⚡ Performance Targets

### Memory Pool Efficiency
- **Reuse Rate**: > 99% (reallocations only on capacity change)
- **Allocation Overhead**: < 1% of compute time
- **Workspace Capacity**: Power-of-2 sizing (128, 256, 512, 1024, ...)

### Kernel Speedups (vs CPU, typical batch size)
| Component | Operation | Batch | CPU | GPU | Target |
|-----------|-----------|-------|-----|-----|--------|
| Attention | Multi-head (512 embed, 8 heads) | 512 | 30ms | 1ms | 30x |
| Feedforward | RichardsGLU (2-layer) | 1K | 50ms | 2ms | 25x |
| SSM | Mamba selective scan | 512 | 40ms | 2ms | 20x |
| Recurrent | RG-LRU forward + backward | 512 | 30ms | 2ms | 15x |

### Numerical Accuracy
- **GPU vs CPU**: < 1e-4 relative error
- **Backward pass**: < 1e-4 gradient tolerance
- **Accumulation**: Use FP32 for all computations (no FP16 unless tested)

---

## 🧪 Testing Patterns

### Unit Test: GPU vs CPU
```rust
#[test]
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn test_attention_gpu_vs_cpu() {
    let input = Array2::random((512, 768));
    let wq = Array2::random((768, 768));
    let wk = Array2::random((768, 768));
    let wv = Array2::random((768, 768));
    let wo = Array2::random((768, 768));
    
    // CPU reference
    let cpu_output = attention_forward_cpu(&input, &wq, &wk, &wv, &wo);
    
    // GPU computation
    let mut kernels = UnifiedGpuKernels::auto_detect().expect("GPU required");
    let params = AttentionParams::new(8, 768, 128, 512);
    let gpu_output = kernels.attention_forward(&input, &wq, &wk, &wv, &wo, &params)
        .expect("GPU forward failed");
    
    // Check numerical equivalence
    for (cpu_val, gpu_val) in cpu_output.iter().zip(gpu_output.iter()) {
        let rel_error = (cpu_val - gpu_val).abs() / (cpu_val.abs().max(1e-6));
        assert!(rel_error < 1e-4, "Relative error {} > 1e-4", rel_error);
    }
}
```

### Integration Test: Component Pipeline
```rust
#[test]
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn test_shared_components_pipeline() {
    let input = Array2::random((512, 768));
    
    // Create GPU-enabled components
    let mut attention = SharedAttentionContext::auto_detect().unwrap();
    let mut feedforward = SharedFeedforward::auto_detect().unwrap();
    let mut temporal = SharedTemporalProcessing::auto_detect().unwrap();
    
    // Pipeline: Attention → Feedforward → Temporal
    let attn_out = attention.forward(&input);
    let ff_out = feedforward.forward(&attn_out);
    let final_out = temporal.forward(&ff_out);
    
    assert_eq!(final_out.shape(), input.shape());
}
```

---

## 🐛 Debugging Checklist

When GPU kernel fails:

1. **Check Feature Flags**
   ```bash
   cargo check --features gpu-cuda
   cargo check --features gpu-metal
   cargo check --features wgpu
   ```

2. **Verify Auto-Detection**
   ```rust
   match GpuDevice::auto_detect() {
       Ok(device) => println!("GPU: {}", device.backend().as_str()),
       Err(e) => println!("No GPU: {}", e),
   }
   ```

3. **Check Workspace Capacity**
   ```rust
   let stats = kernels.workspace_stats();
   println!("Capacity: {:?}", stats.capacity);
   println!("Reallocs: {}", stats.reallocation_count);
   ```

4. **Verify Lock Acquisition**
   - Error message should specify component and method
   - Should NOT say "Failed to acquire GPU device lock" generically

5. **Check Memory Leaks**
   ```rust
   // All allocates should have corresponding deallocates OR workspace reuse
   // Workspace reset should NOT deallocate (buffers reused)
   workspace.reset();  // ✓ Correct: reuse buffers
   // device.deallocate(buf);  // ✗ Wrong: deallocates reusable buffer
   ```

---

## 📋 Checklist: Component GPU Implementation

For each of {AttentionContext, Feedforward, TemporalProcessing}:

- [ ] GPU variant file exists (`*_gpu.rs`)
- [ ] Workspace struct defined with pre-allocated buffers
- [ ] `forward_gpu()` method implemented
- [ ] All operations use workspace pooling (no ad-hoc allocates)
- [ ] Activations computed on GPU (not CPU post-download)
- [ ] Stats tracking: kernel_launches, bytes_uploaded, bytes_downloaded
- [ ] Lock handling: component-specific error messages
- [ ] Backward pass implemented (if needed)
- [ ] Numerical validation test: GPU vs CPU < 1e-4 error
- [ ] Integration test: works in component pipeline
- [ ] Benchmark: latency measured vs CPU baseline
- [ ] Documentation: usage examples, performance targets

---

## 🚀 Workflow: Implementing a GPU Kernel

### Step 1: Create Workspace Structure
```rust
#[derive(Debug)]
struct MyComponentWorkspace {
    buf_input: GpuBuffer,
    buf_weight: GpuBuffer,
    buf_output: GpuBuffer,
    
    capacity: (usize, usize),  // (batch_size, embed_dim)
}

impl MyComponentWorkspace {
    fn ensure_capacity(
        &mut self,
        device: &mut GpuDevice,
        batch_size: usize,
        embed_dim: usize,
    ) -> Result<()> {
        if batch_size <= self.capacity.0 && embed_dim <= self.capacity.1 {
            return Ok(());
        }
        
        let new_batch = batch_size.next_power_of_two();
        let new_embed = embed_dim.next_power_of_two();
        
        self.buf_input = device.allocate(new_batch * new_embed * 4)?;
        self.buf_output = device.allocate(new_batch * new_embed * 4)?;
        // ... other buffers ...
        
        self.capacity = (new_batch, new_embed);
        Ok(())
    }
}
```

### Step 2: Implement GPU Forward
```rust
fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let mut device = self.gpu_device.lock().map_err(|_| ModelError::Backend {
        message: "GPU device lock failed in MyComponent::forward_gpu".to_string(),
    })?;
    
    let (batch, embed) = input.dim();
    let workspace = self.workspace.as_mut().ok_or(ModelError::Backend {
        message: "Workspace not initialized".to_string(),
    })?;
    
    workspace.ensure_capacity(&mut device, batch, embed)?;
    
    // Kernel implementation here
    // ...
    
    Ok(output)
}
```

### Step 3: Add CPU Reference Implementation
```rust
fn forward_gpu_reference_cpu(&self, input: &Array2<f32>) -> Array2<f32> {
    // CPU-only reference for validation
    // Use this for testing GPU vs CPU
}
```

### Step 4: Write Validation Test
```rust
#[test]
fn test_gpu_vs_cpu() {
    let input = Array2::random((128, 512));
    let cpu_output = self.forward_gpu_reference_cpu(&input);
    let gpu_output = self.forward_gpu(&input).unwrap();
    
    assert_relative_error(&cpu_output, &gpu_output, 1e-4);
}
```

### Step 5: Benchmark
```bash
cargo bench --bench my_component_bench --features gpu-all
```

---

## 📞 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "No GPU backend available" | Feature flag not enabled | `cargo build --features gpu-cuda` |
| "GPU device lock failed" | Thread panic during lock | Use proper error mapping |
| "Workspace not initialized" | `workspace` field is None | Call `ensure_capacity()` first |
| "GPU vs CPU output mismatch" | Activation on CPU post-download | Move activation to GPU |
| "Memory usage growing" | Buffers not reused | Use workspace pooling, not allocate/deallocate |
| "Numerical accuracy low" | FP16 precision | Use FP32 for all ops |

---

## 🎯 Priority Order

**Must do first**:
1. Consolidate `GpuDevice::auto_detect()`
2. Standardize lock handling patterns
3. Audit workspace pooling usage

**High priority**:
4. RichardsGLU fused kernel (25x speedup)
5. Attention GPU kernel (30x speedup)
6. Numerical validation tests

**Medium priority**:
7. Mamba selective scan kernel
8. RG-LRU recurrent kernel
9. Performance benchmarks

**Documentation**:
10. Integration guide
11. Best practices document
12. Performance analysis report

---

## 📚 References

- **Thread**: https://ampcode.com/threads/T-019c675f-91bb-7058-b594-cbc0e38d5091
- **Action Plan**: `CONSOLIDATION_GPU_PHASE5_6_ACTION_PLAN.md`
- **Diagnostic**: `GPU_CONSOLIDATION_DIAGNOSTIC_FEB16.md`
- **AGENTS.md**: Build commands, project structure

