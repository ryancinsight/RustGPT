# GPU Consolidation & Performance Optimization Session Plan
**Date**: February 15, 2026  
**Focus**: Continue Phase 5.6.2 GPU kernel integration with automatic GPU detection (no fallback)  
**Status**: Ready for Implementation  

---

## Executive Summary

This session continues Phase 5.6.2 implementation by:
1. **Integrating RichardsGLU fused kernel** into the component (kernel is implemented)
2. **Implementing GPU backward pass** for gradient computation
3. **Creating numerical validation tests** (GPU vs CPU reference)
4. **Implementing automatic GPU detection** with strict no-fallback semantics
5. **Optimizing memory efficiency** with zero-copy pipeline

**Key Constraint**: No CPU fallback - errors must propagate if GPU is unavailable.

---

## Current Infrastructure Status

### ✅ Completed (Phase 5.6.1)
- `GpuComponent` trait for all shared layers
- `GpuDevice` abstraction with unified interface
- `UnifiedGpuBufferPool` for memory management
- WGPU fused kernel shader for RichardsGLU (515-591 in wgpu_ops.rs)
- RichardsGLU fused kernel method (4237-4400+ in wgpu_ops.rs)
- All compilation warnings fixed

### 🔄 In Progress
- Integration of `forward_gpu_fused()` into RichardsGlu component
- GPU backward pass implementation
- Numerical validation test suite
- Automatic GPU detection without CPU fallback

### 📋 Outstanding
- PolyAttention GPU kernels
- Mamba/SSM GPU kernels
- RG-LRU GPU kernels
- Performance benchmarking

---

## Phase 5.6.2 Execution Plan

### WORKSTREAM 1: RichardsGLU GPU Forward & Backward (PRIORITY 1)

#### Objective
Integrate fused kernel into RichardsGlu component with full forward/backward pass.

#### Step 1.1: Add GPU Forward Method to RichardsGlu
**File**: `src/domain/richards/richards_glu.rs`

```rust
impl RichardsGlu {
    /// GPU-accelerated forward pass using fused kernel
    pub fn forward_gpu_fused(
        &self,
        input: &Array2<f32>,
        device: &Arc<Mutex<GpuDevice>>,
        pool: &mut dyn GpuMemoryPool,
    ) -> Result<Array2<f32>> {
        // 1. Lock device and get WgpuMatrixOps
        let dev = device.lock().map_err(|_| Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to lock GPU device"
        )))?;
        
        // 2. Convert CPU tensors to GPU buffers
        let input_gpu = pool.allocate_f32(input.len())?;
        pool.copy_host_to_device(input.as_ptr() as *const _, input_gpu.id, input.len())?;
        
        // 3. Get weight buffers
        let w1_gpu = self.w1_gpu_buffer(pool)?;
        let w2_gpu = self.w2_gpu_buffer(pool)?;
        let w_out_gpu = self.w_out_gpu_buffer(pool)?;
        
        // 4. Allocate output buffer
        let batch_size = input.nrows();
        let output_size = batch_size * self.w_out.ncols();
        let mut output_gpu = pool.allocate_f32(output_size)?;
        
        // 5. Create parameters struct
        let params = RichardsGluFusedParams {
            batch_size: batch_size as u32,
            input_dim: input.ncols() as u32,
            hidden_dim: self.w1.ncols() as u32,
            output_dim: self.w_out.ncols() as u32,
            nu: self.richards_activation.nu,
            k: self.richards_activation.k,
            m: self.richards_activation.m,
            beta: self.richards_activation.beta,
            temp_reciprocal: 1.0 / (self.richards_activation.temperature + 1e-8),
            gate_scale: self.gate.scale,
            gate_bias: self.gate.bias,
            gate_temp_reciprocal: 1.0 / (self.gate.temperature + 1e-8),
            value_scale: 1.0,
            output_gain: 1.0,
            _pad1: 0,
            _pad2: 0,
        };
        
        // 6. Call fused kernel
        ops.richards_glu_fused(
            pool,
            &input_gpu,
            &w1_gpu,
            &w2_gpu,
            &w_out_gpu,
            &mut output_gpu,
            batch_size,
            input.ncols(),
            self.w1.ncols(),
            self.w_out.ncols(),
            params,
        )?;
        
        // 7. Download result from GPU
        let mut result = Array2::zeros((batch_size, self.w_out.ncols()));
        pool.copy_device_to_host(output_gpu.id, result.as_mut_ptr() as *mut _, output_size)?;
        
        Ok(result)
    }
    
    /// Numerical validation: GPU output vs CPU
    pub fn validate_gpu_accuracy(
        &self,
        input: &Array2<f32>,
        device: &Arc<Mutex<GpuDevice>>,
        pool: &mut dyn GpuMemoryPool,
        tolerance: f32,
    ) -> Result<bool> {
        // 1. Compute CPU reference
        let cpu_output = self.forward(input);
        
        // 2. Compute GPU output
        let gpu_output = self.forward_gpu_fused(input, device, pool)?;
        
        // 3. Compare with tolerance
        let max_diff = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        
        Ok(max_diff <= tolerance)
    }
}
```

#### Step 1.2: GPU Backward Pass
**File**: `src/domain/richards/richards_glu.rs`

```rust
impl RichardsGlu {
    /// GPU-accelerated backward pass (gradient computation)
    pub fn backward_gpu(
        &mut self,
        input: &Array2<f32>,
        loss_gradient: &Array2<f32>,
        device: &Arc<Mutex<GpuDevice>>,
        pool: &mut dyn GpuMemoryPool,
    ) -> Result<Array2<f32>> {
        // Gradient computation:
        // ∂L/∂W_out = gated^T @ loss_grad
        // ∂L/∂gated = loss_grad @ W_out^T
        // ∂L/∂value = ∂L/∂gated * gate
        // ∂L/∂gate = ∂L/∂gated * value
        // ∂L/∂x1 = ∂L/∂value * richards'(x1) * (1 + x1 * richards''(x1))
        // ∂L/∂x2 = ∂L/∂gate * richards'(x2)
        // ∂L/∂W1 = input^T @ x1_grad
        // ∂L/∂W2 = input^T @ x2_grad
        // ∂L/∂input = x1_grad @ W1^T + x2_grad @ W2^T
        
        let batch_size = input.nrows();
        let input_dim = input.ncols();
        let hidden_dim = self.w1.ncols();
        
        // 1. Allocate gradient buffers
        let mut grad_input = pool.allocate_f32(batch_size * input_dim)?;
        let mut grad_w1 = pool.allocate_f32(input_dim * hidden_dim)?;
        let mut grad_w2 = pool.allocate_f32(input_dim * hidden_dim)?;
        let mut grad_w_out = pool.allocate_f32(hidden_dim * self.w_out.ncols())?;
        
        // 2. Upload loss gradient to GPU
        let loss_grad_gpu = pool.allocate_f32(loss_gradient.len())?;
        pool.copy_host_to_device(
            loss_gradient.as_ptr() as *const _,
            loss_grad_gpu.id,
            loss_gradient.len(),
        )?;
        
        // 3. Call GPU backward kernel
        // (Would implement custom backward kernel in WGSL)
        
        // 4. Download gradients
        let mut grad_w1_out = vec![0.0; input_dim * hidden_dim];
        pool.copy_device_to_host(
            grad_w1.id,
            grad_w1_out.as_mut_ptr(),
            grad_w1_out.len(),
        )?;
        
        // 5. Update weights with optimizer
        let grad_w1_array = Array2::from_shape_vec((input_dim, hidden_dim), grad_w1_out)?;
        self.optimizer_w1.update(&mut self.w1, &grad_w1_array)?;
        
        // Similar for W2 and W_out...
        
        let mut grad_input_out = vec![0.0; batch_size * input_dim];
        pool.copy_device_to_host(
            grad_input.id,
            grad_input_out.as_mut_ptr(),
            grad_input_out.len(),
        )?;
        
        Ok(Array2::from_shape_vec((batch_size, input_dim), grad_input_out)?)
    }
}
```

#### Success Criteria
- [ ] `forward_gpu_fused()` compiles and executes without fallback
- [ ] GPU output matches CPU within ε ≤ 1e-4
- [ ] All batch sizes (1, 32, 128, 512, 1024) validated
- [ ] Backward pass computes gradients correctly
- [ ] Weight updates reflected in iterative calls

---

### WORKSTREAM 2: Numerical Validation Tests (PRIORITY 1)

#### Objective
Create comprehensive test suite validating GPU operations against CPU reference.

#### File: `tests/gpu_richards_glu_validation.rs`

```rust
#[cfg(test)]
mod gpu_richards_glu_validation {
    use ndarray::Array2;
    use std::sync::{Arc, Mutex};
    
    #[test]
    fn test_richardson_glu_gpu_vs_cpu_accuracy() {
        let input = Array2::from_elem((32, 512), 0.5f32);
        let mut glu = RichardsGlu::new(512, 1024, 512);
        
        // CPU forward pass
        let cpu_output = glu.forward(&input);
        
        // GPU forward pass
        let device = Arc::new(Mutex::new(GpuDevice::auto_detect().unwrap()));
        let mut pool = UnifiedGpuBufferPool::new(1024 * 1024 * 100); // 100MB
        let gpu_output = glu.forward_gpu_fused(&input, &device, &mut pool).unwrap();
        
        // Validate accuracy
        assert!(glu.validate_gpu_accuracy(&input, &device, &mut pool, 1e-4).unwrap());
    }
    
    #[test]
    fn test_richardson_glu_gpu_batch_sizes() {
        let mut glu = RichardsGlu::new(512, 1024, 512);
        let device = Arc::new(Mutex::new(GpuDevice::auto_detect().unwrap()));
        let mut pool = UnifiedGpuBufferPool::new(1024 * 1024 * 100);
        
        for batch_size in &[1, 32, 128, 512, 1024] {
            let input = Array2::from_elem((*batch_size, 512), 0.5f32);
            assert!(glu.validate_gpu_accuracy(&input, &device, &mut pool, 1e-4).unwrap());
        }
    }
    
    #[test]
    fn test_richardson_glu_gpu_backward_pass() {
        let input = Array2::from_elem((32, 512), 0.5f32);
        let loss_grad = Array2::from_elem((32, 512), 0.01f32);
        let mut glu = RichardsGlu::new(512, 1024, 512);
        
        let device = Arc::new(Mutex::new(GpuDevice::auto_detect().unwrap()));
        let mut pool = UnifiedGpuBufferPool::new(1024 * 1024 * 100);
        
        // Store initial weights
        let w1_before = glu.w1.clone();
        
        // Backward + update
        let _grad_input = glu.backward_gpu(&input, &loss_grad, &device, &mut pool).unwrap();
        
        // Weights should have changed
        assert_ne!(glu.w1.as_ptr(), w1_before.as_ptr());
        assert!((glu.w1 - w1_before).norm_max() > 0.0);
    }
}
```

#### Success Criteria
- [ ] All batch size tests pass
- [ ] Accuracy tolerance ε ≤ 1e-4 maintained
- [ ] Backward pass gradient computation correct
- [ ] Zero memory leaks in tests

---

### WORKSTREAM 3: Automatic GPU Detection (PRIORITY 2)

#### Objective
Implement unified GPU detection with strict no-fallback semantics.

#### File: `src/domain/compute/gpu_device.rs`

```rust
impl GpuDevice {
    /// Automatic GPU detection with strict no-fallback
    /// Errors if GPU unavailable - does not fall back to CPU
    pub fn auto_detect() -> Result<Self> {
        // 1. Detect available backends
        let mut backends = Vec::new();
        
        #[cfg(feature = "gpu-wgpu")]
        {
            match Self::init_wgpu() {
                Ok(dev) => backends.push(("WGPU", dev)),
                Err(e) => eprintln!("WGPU init failed: {}", e),
            }
        }
        
        #[cfg(feature = "gpu-cuda")]
        {
            match Self::init_cuda() {
                Ok(dev) => backends.push(("CUDA", dev)),
                Err(e) => eprintln!("CUDA init failed: {}", e),
            }
        }
        
        #[cfg(feature = "gpu-metal")]
        {
            match Self::init_metal() {
                Ok(dev) => backends.push(("Metal", dev)),
                Err(e) => eprintln!("Metal init failed: {}", e),
            }
        }
        
        // 2. Check if any backend succeeded
        if backends.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No GPU backends available. Ensure one of: gpu-wgpu, gpu-cuda, gpu-metal features is enabled"
            )));
        }
        
        // 3. Report available backends and select first
        eprintln!("GPU backends available: {:?}", backends.iter().map(|(n, _)| n).collect::<Vec<_>>());
        Ok(backends.into_iter().next().unwrap().1)
    }
    
    /// Strict validation: returns error if GPU not available
    pub fn validate_availability(&self) -> Result<()> {
        match &self.backend {
            GpuBackend::Cpu => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "CPU backend detected - GPU required. Enable gpu-wgpu, gpu-cuda, or gpu-metal"
            ))),
            _ => Ok(()),
        }
    }
}
```

#### Success Criteria
- [ ] `auto_detect()` succeeds with GPU enabled
- [ ] `auto_detect()` errors clearly if GPU unavailable
- [ ] No silent CPU fallback
- [ ] Error message guides users to enable GPU features

---

### WORKSTREAM 4: Memory Efficiency Optimization (PRIORITY 3)

#### Objective
Implement zero-copy pipeline for full forward pass on GPU.

#### Steps
1. **Buffer pre-allocation** (all buffers allocated once at startup)
2. **Power-of-2 sizing** (reduce fragmentation)
3. **Reuse tracking** (monitor allocation efficiency)
4. **Zero-copy forwarding** (input→GPU, stays on GPU for all layers, output→CPU once)

#### File: `src/domain/compute/unified_gpu_buffer_pool.rs`

```rust
impl UnifiedGpuBufferPool {
    pub fn allocate_with_reuse(&mut self, size: usize) -> Result<GpuBuffer> {
        // Find power-of-2 fitting size
        let pow2_size = (size as u64).next_power_of_two() as usize;
        
        // Check for reusable buffer
        for (i, buf) in self.buffers.iter().enumerate() {
            if buf.capacity >= pow2_size && !self.in_use.contains(&buf.id) {
                self.reuse_count += 1;
                self.in_use.insert(buf.id);
                return Ok(buf.clone());
            }
        }
        
        // Allocate new if no reusable buffer found
        let new_buf = self.allocate_f32(pow2_size)?;
        self.total_allocated += pow2_size * 4; // 4 bytes per f32
        Ok(new_buf)
    }
    
    pub fn get_efficiency_stats(&self) -> EfficiencyStats {
        let wasted = self.buffers
            .iter()
            .map(|b| b.capacity - b.used)
            .sum::<usize>() * 4;
        
        EfficiencyStats {
            total_allocated: self.total_allocated,
            total_wasted_padding: wasted,
            reuse_count: self.reuse_count,
            resize_count: self.resize_count,
            efficiency: (self.total_allocated - wasted) as f32 / self.total_allocated as f32,
        }
    }
}
```

#### Success Criteria
- [ ] Buffer efficiency > 92%
- [ ] Reuse count > 100 for 1000 forward passes
- [ ] Resize count < 5 for 1000 passes
- [ ] Zero-copy pipeline reduces transfer latency by 50%+

---

## Implementation Order

### Session Phase 1 (4 hours)
1. ✅ Cleanup compilation warnings (DONE)
2. Add GPU forward method to RichardsGlu
3. Create numerical validation test
4. Run validation tests on all batch sizes

### Session Phase 2 (3 hours)
1. Implement GPU backward pass
2. Implement automatic GPU detection
3. Add error handling for no-fallback semantics
4. Test GPU detection with/without GPU

### Session Phase 3 (2 hours)
1. Optimize memory efficiency
2. Profile zero-copy pipeline
3. Benchmark performance (target: 25x speedup)
4. Document results

---

## Compilation Commands

```bash
# Check compilation
cargo check --lib --features gpu-wgpu

# Build with WGPU
cargo build --release --features gpu-wgpu

# Run GPU validation tests
cargo test --lib gpu_richards_glu_validation --features gpu-wgpu

# Run detection tests
cargo test --lib gpu_detection --features gpu-wgpu

# Benchmark
cargo bench --bench gpu_performance --features gpu-wgpu
```

---

## Key Files to Modify

| File | Purpose | Lines |
|------|---------|-------|
| `src/domain/richards/richards_glu.rs` | Add GPU forward/backward | +150 |
| `tests/gpu_richards_glu_validation.rs` | Validation test suite | +200 |
| `src/domain/compute/gpu_device.rs` | Auto-detection + no-fallback | +80 |
| `src/domain/compute/unified_gpu_buffer_pool.rs` | Memory efficiency | +60 |

---

## Success Metrics

### Performance
- [ ] RichardsGLU: 25x speedup on 1K batch
- [ ] GPU memory usage: <200MB for 1K batch
- [ ] Latency: <2ms per forward pass

### Correctness
- [ ] GPU vs CPU max difference: ≤ 1e-4
- [ ] All batch sizes (1-1024) validated
- [ ] Backward pass gradient accuracy: ≤ 1e-3

### Reliability
- [ ] Zero-copy pipeline achieves 100% GPU residence for forward
- [ ] Memory efficiency >92%
- [ ] All tests pass without GPU fallback

---

## Risk Mitigation

| Risk | Mitigation | Validation |
|------|-----------|-----------|
| Numerical instability | Test all batch sizes, compare vs CPU | Tolerance ε ≤ 1e-4 |
| Memory fragmentation | Power-of-2 sizing, pre-allocation | Efficiency >92% |
| GPU unavailable | Strict error handling, no fallback | Error test passes |
| Kernel correctness | Atomic operations, thread safety | Deterministic output |

---

## Next Session Handoff

**If completed**:
- RichardsGLU GPU forward/backward fully working
- 25x speedup verified on large batches
- All validation tests passing
- GPU detection robust with clear error messages

**Next priorities**:
1. PolyAttention GPU kernels
2. Mamba/SSM GPU kernels  
3. RG-LRU GPU kernels
4. Full integration benchmarks

---

**Prepared**: February 15, 2026  
**Expected Duration**: 9-12 hours  
**Status**: Ready for implementation
