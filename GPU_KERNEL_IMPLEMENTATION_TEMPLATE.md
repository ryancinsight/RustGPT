# GPU Kernel Implementation Template
## Step-by-Step Guide for Adding New GPU Operations

**For**: WGPU, CUDA, Metal backends  
**Date**: 2026-02-16

---

## Quick Template: Add New GPU Operation

### Step 1: Define WGSL Shader (wgsl_kernels.rs)

```rust
//! In src/domain/compute/wgsl_kernels.rs

pub const SHADER_MY_OPERATION: &str = r#"
// Uniform parameters
struct MyOperationParams {
    param1: f32,
    param2: u32,
    pad1: u32,
    pad2: u32,
}

// Input/Output buffers
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: MyOperationParams;

// Compute kernel
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    // Your kernel logic here
    output[idx] = process_element(input[idx], params);
}
"#;
```

### Step 2: Add Trait Method (gpu_ops.rs)

```rust
//! In src/domain/compute/gpu_ops.rs

pub trait GpuMatrixOps: Send + Sync {
    /// My operation description
    fn my_operation(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;
}

// CPU stub implementation (returns error)
impl GpuMatrixOps for CpuGpuMatrixOps {
    fn my_operation(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "CPU my_operation not implemented".to_string(),
        })
    }
}
```

### Step 3: Implement for WGPU (wgpu_ops.rs)

```rust
//! In src/domain/compute/wgpu_ops.rs

#[cfg(feature = "wgpu")]
impl GpuMatrixOps for WgpuMatrixOps {
    fn my_operation(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        // 1. Create compute pipeline from WGSL shader
        let pipeline = self
            .get_or_create_pipeline(
                "my_operation",
                super::super::wgsl_kernels::SHADER_MY_OPERATION,
            )
            .map_err(|e| ModelError::Backend {
                message: format!("Failed to create pipeline: {}", e),
            })?;

        // 2. Create bind group with buffers
        let bind_group = self.device.create_bind_group(
            "my_operation_bg",
            &pipeline.get_bind_group_layout(0),
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_wgpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_wgpu_buffer_mut().as_entire_binding(),
                },
            ],
        );

        // 3. Submit compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch workgroups: 256 threads per workgroup
            let workgroups = (size + 255) / 256;
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }
}
```

### Step 4: Implement for CUDA (cuda/ops.rs)

```rust
//! In src/domain/compute/cuda/ops.rs

#[cfg(feature = "gpu-cuda")]
impl GpuMatrixOps for CudaMatrixOps {
    fn my_operation(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        // Load or compile kernel
        let kernel = self.get_or_load_kernel("my_operation.cu")?;
        
        // Set up kernel parameters
        let grid_size = (size + 255) / 256;
        
        // Launch kernel
        unsafe {
            kernel.launch_on_stream(
                self.stream,
                [grid_size, 1, 1],
                [256, 1, 1],
                0,
                &[
                    input.as_cudarc_device_ptr().as_void(),
                    output.as_cudarc_device_ptr_mut().as_void(),
                    &size as *const _ as *const std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }
}
```

### Step 5: Implement for Metal (metal/ops.rs)

```rust
//! In src/domain/compute/metal/ops.rs

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
impl GpuMatrixOps for MetalMatrixOps {
    fn my_operation(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        // Compile/load Metal shader
        let library = self.device.new_library_with_source(
            METAL_MY_OPERATION_KERNEL,
            Default::default(),
        )?;
        
        let function = library.get_function("my_operation_kernel", None)?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)?;

        // Create command buffer and encoder
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input.as_metal_buffer()), 0);
        encoder.set_buffer(1, Some(&output.as_metal_buffer_mut()), 0);

        // Dispatch threads
        let threads_per_group = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = metal::MTLSize {
            width: (size + 255) / 256,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(groups, threads_per_group);
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        Ok(())
    }
}
```

### Step 6: Add High-Level Wrapper (gpu_device.rs)

```rust
//! In src/domain/compute/gpu_device.rs

impl GpuDevice {
    /// My operation wrapper
    pub fn my_operation(
        &mut self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        self.ops.my_operation(self.memory.as_mut(), input, output, size)
    }
}
```

### Step 7: Add High-Level Wrapper (unified_gpu_kernels.rs)

```rust
//! In src/domain/layers/components/unified_gpu_kernels.rs

impl UnifiedGpuKernels {
    /// My operation forward pass
    pub fn my_operation_forward(
        &mut self,
        input: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock".to_string(),
        })?;

        let (batch_size, dim) = input.dim();

        // Allocate GPU buffers
        let input_size = batch_size * dim * std::mem::size_of::<f32>();
        let mut input_buf = device.allocate(input_size)?;
        let mut output_buf = device.allocate(input_size)?;

        // Upload input
        device.upload(input.as_slice().unwrap(), &mut input_buf)?;

        // Execute operation
        device.my_operation(&input_buf, &mut output_buf, batch_size * dim)?;

        // Download output
        let mut output = vec![0.0f32; batch_size * dim];
        device.download(&output_buf, &mut output)?;

        // Cleanup
        device.deallocate(input_buf);
        device.deallocate(output_buf);

        // Reshape to Array2
        Ok(Array2::from_shape_vec((batch_size, dim), output)?)
    }
}
```

### Step 8: Add Unit Tests

```rust
//! Tests in same module

#[test]
fn test_my_operation_correctness() {
    // Test on CPU reference
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let expected = cpu_my_operation(&input);
    
    assert_eq!(expected, vec![...]);
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[test]
fn test_my_operation_gpu() {
    if let Ok(mut device) = GpuDevice::auto_detect() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let size = input_data.len();

        // Allocate buffers
        let mut input_buf = device.allocate_f32(size).unwrap();
        let mut output_buf = device.allocate_f32(size).unwrap();

        // Upload and execute
        device.upload(&input_data, &mut input_buf).unwrap();
        device.my_operation(&input_buf, &mut output_buf, size).unwrap();

        // Download and verify
        let mut output = vec![0.0; size];
        device.download(&output_buf, &mut output).unwrap();

        let expected = cpu_my_operation(&input_data);
        for i in 0..size {
            assert!((output[i] - expected[i]).abs() < 1e-4);
        }

        device.deallocate(input_buf);
        device.deallocate(output_buf);
    }
}
```

---

## Checklist for New GPU Operation

- [ ] WGSL shader implemented in `wgsl_kernels.rs`
- [ ] Trait method added to `GpuMatrixOps` in `gpu_ops.rs`
- [ ] CPU stub returns informative error
- [ ] WGPU implementation in `wgpu_ops.rs`
- [ ] CUDA stub in `cuda/ops.rs` (ready for `.cu` kernel)
- [ ] Metal stub in `metal/ops.rs` (ready for `.metal` kernel)
- [ ] High-level wrapper in `gpu_device.rs`
- [ ] Kernel dispatcher method in `unified_gpu_kernels.rs`
- [ ] Unit tests (correctness + GPU)
- [ ] Documentation with examples

---

## Common Mistakes to Avoid

### ❌ Silent CPU Fallback
```rust
// DON'T DO THIS
fn my_operation(...) -> Result<()> {
    match gpu_kernel(...) {
        Ok(()) => Ok(()),
        Err(_) => {
            cpu_my_operation(...);  // ❌ FALLBACK
            Ok(())
        }
    }
}
```

### ✅ Explicit GPU-Only
```rust
// DO THIS
fn my_operation(...) -> Result<()> {
    gpu_kernel(...)?  // Propagate error
}
```

### ❌ Forgetting Workgroup Barrier
```wgsl
// DON'T DO THIS (race condition)
shared_data[tid] = input[idx];
// immediately use shared_data[other_tid]  // ❌ Other threads may not have written
```

### ✅ Proper Synchronization
```wgsl
// DO THIS
shared_data[tid] = input[idx];
workgroupBarrier();  // Wait for all threads
// Now all shared_data is valid
```

### ❌ Unbounded Memory Allocation
```wgsl
// DON'T DO THIS
var<workgroup> huge: array<f32, 100000>;  // ❌ > 48KB limit
```

### ✅ Bounded Shared Memory
```wgsl
// DO THIS
var<workgroup> small: array<f32, 256>;  // ✅ 1KB per workgroup
```

---

## Backend-Specific Notes

### WGPU
- Portable across Vulkan, Metal, DX12
- Pipeline caching for repeated kernels
- Bind group layout must match shader
- Uses thread pools for queue submission

### CUDA
- Best for NVIDIA devices (V100+, A100, H100)
- Requires `.cu` kernel files
- Compile-time or runtime compilation
- Direct memory pointer management

### Metal
- macOS/iOS only (Apple Silicon, Intel discrete)
- Metal Performance Shaders for standard ops
- Metal Compute kernels for custom kernels
- iOS requires different deployment target

---

## Performance Debugging

### Check Kernel Dispatch
```rust
// Ensure workgroups cover all data
let workgroups = (size + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;
// NOT: size / THREADS_PER_GROUP (misses last few elements if not aligned)
```

### Verify Data Transfer
```rust
// Validate memory sizes
let buf_size_bytes = num_elements * std::mem::size_of::<f32>();
device.allocate(buf_size_bytes)?;  // Must match
```

### Test Numerical Accuracy
```rust
// Compare GPU vs CPU
let epsilon = 1e-4;
for i in 0..output.len() {
    assert!((gpu_output[i] - cpu_output[i]).abs() < epsilon,
        "Mismatch at index {}: gpu={}, cpu={}",
        i, gpu_output[i], cpu_output[i]
    );
}
```

