# Quick Reference: GPU Backend Implementation

## Current Status
- **WGPU**: ✅ FULLY IMPLEMENTED (cross-platform)
- **CUDA**: ⚠️ STUB (returns errors, ready for .cu kernel implementation)
- **Metal**: ⚠️ STUB (returns errors, ready for .metal kernel implementation)

## Add a New GPU Operation

### 1. Add Trait Method to `gpu_ops.rs`

```rust
pub trait GpuMatrixOps: Send + Sync {
    /// Your operation description
    ///
    /// # Arguments
    /// * `pool` - Memory pool for buffer access
    /// * ... (other params)
    fn your_operation(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &YourParams,
        size: usize,
    ) -> Result<()>;
}
```

### 2. Implement in WGPU (`wgpu_ops.rs`)

**a) Add shader:**
```wgsl
const SHADER_YOUR_OP: &str = r#"
struct YourParams {
    param1: f32,
    param2: f32,
    // ...
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: YourParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&output)) {
        return;
    }
    
    // Your computation
    output[i] = input[i] + params.param1;
}
"#;
```

**b) Add implementation method:**
```rust
fn your_operation(
    &mut self,
    pool: &mut dyn GpuMemoryPool,
    input: &GpuBuffer,
    output: &mut GpuBuffer,
    params: &YourParams,
    size: usize,
) -> Result<()> {
    let buf_in = Self::resolve_buffer(pool, input.id)?;
    let buf_out = Self::resolve_buffer(pool, output.id)?;

    let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("YourOp Params"),
        contents: bytemuck::cast_slice(&[*params]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let pipeline = self.get_or_create_pipeline(
        "your_op",
        SHADER_YOUR_OP,
        &[
            // Bind group layout entries...
        ],
    )?;

    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("YourOp Bind Group"),
        layout: &self.bind_group_layouts["your_op"],
        entries: &[
            // Bind group entries...
        ],
    });

    let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("YourOp Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("YourOp Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (size as u32 + 255) / 256;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    self.queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
```

### 3. Implement in CUDA (`cuda/ops.rs`)

For now, add stub that returns error:
```rust
fn your_operation(
    &mut self,
    _pool: &mut dyn GpuMemoryPool,
    _input: &GpuBuffer,
    _output: &mut GpuBuffer,
    _params: &YourParams,
    size: usize,
) -> Result<()> {
    Err(ModelError::Backend {
        message: format!(
            "CUDA your_operation not yet implemented for size {}. \
             Use WGPU backend or compile with native CUDA kernels.",
            size
        ),
    })
}
```

### 4. Implement in Metal (`metal/ops.rs`)

Same stub as CUDA for now:
```rust
fn your_operation(
    &mut self,
    _pool: &mut dyn GpuMemoryPool,
    _input: &GpuBuffer,
    _output: &mut GpuBuffer,
    _params: &YourParams,
    size: usize,
) -> Result<()> {
    Err(ModelError::Backend {
        message: format!(
            "Metal your_operation not yet implemented for size {}. \
             Use WGPU backend or compile with native Metal kernels.",
            size
        ),
    })
}
```

### 5. Use in Your Code

```rust
use crate::domain::compute::GpuDevice;

let mut device = GpuDevice::auto_detect()?;  // CUDA → Metal → WGPU
device.your_operation(&mut pool, &input, &mut output, &params, size)?;
```

## WGSL Shader Template

```wgsl
// 1. Define parameters struct
struct YourParams {
    scale: f32,
    bias: f32,
    threshold: f32,
    pad: u32,  // Align to 16 bytes
}

// 2. Define storage/uniform buffers
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: YourParams;

// 3. Optional: shared memory (fast cache)
var<workgroup> shared_data: array<f32, 256>;

// 4. Compute entry point
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let global_idx = global_id.x;
    let local_idx = local_id.x;
    
    // Bounds check
    if (global_idx >= arrayLength(&output)) {
        return;
    }
    
    // Load from global memory
    let value = input[global_idx];
    
    // Compute (can use shared memory for cross-thread operations)
    let result = params.scale * value + params.bias;
    
    // Store to global memory
    output[global_idx] = result;
}
```

## WGPU Bind Group Layout Template

```rust
let pipeline = self.get_or_create_pipeline(
    "my_op",
    SHADER_MY_OP,
    &[
        // Input buffer (read-only storage)
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // Output buffer (read-write storage)
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // Parameters (uniform buffer)
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ],
)?;
```

## Testing Your GPU Operation

```rust
#[test]
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn test_my_gpu_operation() {
    if let Ok(mut device) = crate::domain::compute::GpuDevice::auto_detect() {
        let size = 1024;
        let input = device.allocate_f32(size).unwrap();
        let mut output = device.allocate_f32(size).unwrap();
        
        // Upload test data
        let input_data = vec![1.0f32; size];
        device.upload(&input_data, &mut input.clone()).unwrap();
        
        // Run operation
        let params = YourParams { /* ... */ };
        device.your_operation(&mut pool, &input, &mut output, &params, size)
            .expect("GPU operation failed");
        
        // Download and verify
        let mut output_data = vec![0.0f32; size];
        device.download(&output, &mut output_data).unwrap();
        
        // Assertions
        for &val in &output_data {
            assert!(val.is_finite(), "Output should be finite");
        }
        
        // Cleanup
        device.deallocate(input);
        device.deallocate(output);
    } else {
        println!("No GPU available, skipping test");
    }
}
```

## Debugging GPU Operations

### Enable WGPU Validation
```bash
WGPU_BACKEND=vulkan WGPU_POWER_PREFERENCE=high_performance cargo test --lib
```

### Check GPU Detection
```rust
match crate::domain::compute::ComputeBackend::detect_available_gpu_backends() {
    backends if !backends.is_empty() => {
        println!("Available backends: {:?}", backends);
    }
    _ => println!("No GPU backends available"),
}
```

### Verify Operation
```bash
cargo test --lib "test_my_gpu_operation" -- --nocapture
```

## Performance Tips

1. **Coalesce Memory Access**: Access memory sequentially in kernel
2. **Use Workgroup Shared Memory**: For small temp data (< 48KB)
3. **Dispatch Efficiently**: Use (size + 255) / 256 workgroups for 256-thread groups
4. **Minimize Kernel Launches**: Fuse operations when possible
5. **Profile with Actual Data**: Test with real input dimensions

## Common Errors

| Error | Fix |
|-------|-----|
| "Operation not yet implemented" | Use WGPU or implement native CUDA/Metal kernel |
| "Bounds check failed" | Add bounds checking in shader before array access |
| "Buffer not found" | Call `Self::resolve_buffer()` on all GPU buffers |
| "Shader compilation failed" | Check WGSL syntax in shader constants |
| "Resource not bound" | Verify bind group layout matches shader bindings |

## Key Files

- `src/domain/compute/gpu_ops.rs` - Trait definition
- `src/domain/compute/wgpu_ops.rs` - WGPU implementation
- `src/domain/compute/cuda/ops.rs` - CUDA stubs
- `src/domain/compute/metal/ops.rs` - Metal stubs
- `src/domain/compute/gpu_device.rs` - Device abstraction
