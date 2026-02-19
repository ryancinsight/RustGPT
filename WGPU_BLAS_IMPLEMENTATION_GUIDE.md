# WGPU BLAS Implementation Guide
**Purpose**: Quick reference for implementing GPU BLAS operations  
**Target**: Phase 5.5 GPU consolidation

---

## 1. Project Structure

```
src/domain/compute/
├── gpu_ops.rs                     # Trait definitions (GpuMatrixOps)
├── wgpu_ops.rs                    # Main WGPU implementation
├── gpu_device.rs                  # High-level GPU interface (uses wgpu_ops)
├── gpu_memory.rs                  # Memory pool interface
├── wgpu_shaders/                  # WGSL shader files (NEW)
│   ├── gemm.wgsl                  # Matrix multiply
│   ├── activation.wgsl             # Element-wise ops
│   ├── softmax.wgsl               # Normalization
│   ├── attention.wgsl             # PolyAttention kernels
│   └── utility.wgsl               # Common functions
└── tests/
    └── gpu_blas_*.rs              # Validation tests
```

---

## 2. WGPU Shader Compilation Pattern

### Shader Loading in Rust
```rust
// In wgpu_ops.rs
use wgpu::{Device, Queue, ShaderSource};

fn create_compute_pipeline(
    device: &wgpu::Device,
    shader_source: &str,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry_point),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{} layout", entry_point)),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&layout),
        module: &module,
        entry_point,
    })
}
```

### Inline Shader Macro (Recommended)
```toml
[dependencies]
wgsl-inline = "0.3"
```

```rust
use wgsl_inline::wgsl;

let shader = wgsl!(
    r#"
    @compute @workgroup_size(16, 16)
    fn gemm_kernel(...) {
        // Implementation
    }
    "#
);
```

---

## 3. GEMM (General Matrix Multiply) Implementation

### Algorithm: Tile-based GEMM
```
Goal: Compute C = alpha * (A @ B) + beta * C
Where A is M×K, B is K×N, C is M×N

Approach:
1. Divide output into tiles (e.g., 16×16 blocks)
2. Each workgroup computes one output tile
3. Each thread in workgroup computes one element
4. Loop over K-dimension, accumulating partial products
5. Use shared memory for tile of A and B (reduce global memory reads)
```

### WGSL Kernel Template

```wgsl
struct Params {
    m: u32,      // Rows of A (and C)
    n: u32,      // Cols of B (and C)
    k: u32,      // Cols of A, Rows of B
    alpha: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn gemm_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let i = global_id.x;  // Row in C
    let j = global_id.y;  // Col in C
    
    if (i >= params.m || j >= params.n) {
        return;
    }
    
    // Accumulate A[i, :] @ B[:, j]
    var sum: f32 = 0.0;
    for (var kk: u32 = 0u; kk < params.k; kk += 1u) {
        let a_idx = i * params.k + kk;
        let b_idx = kk * params.n + j;
        sum += a[a_idx] * b[b_idx];
    }
    
    // Write result: C[i, j] = alpha * sum + beta * C[i, j]
    let c_idx = i * params.n + j;
    c[c_idx] = params.alpha * sum + params.beta * c[c_idx];
}
```

### Optimized Tile-based Version (Better Performance)
```wgsl
// Shared memory for tile of A and B
var<workgroup> a_tile: array<array<f32, 16>, 16>;
var<workgroup> b_tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn gemm_tiled(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let i_local = local_id.x;
    let j_local = local_id.y;
    
    let i_global = wg_id.x * 16u + i_local;
    let j_global = wg_id.y * 16u + j_local;
    
    if (i_global >= params.m || j_global >= params.n) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Tile over K dimension
    for (var k_tile: u32 = 0u; k_tile < params.k; k_tile += 16u) {
        // Load A[i_global, k_tile + j_local]
        a_tile[i_local][j_local] = a[(i_global * params.k) + (k_tile + j_local)];
        
        // Load B[k_tile + i_local, j_global]
        b_tile[i_local][j_local] = b[((k_tile + i_local) * params.n) + j_global];
        
        workgroupBarrier();
        
        // Accumulate: sum += A[i_local, :] * B[:, j_local]
        for (var k: u32 = 0u; k < 16u; k += 1u) {
            sum += a_tile[i_local][k] * b_tile[k][j_local];
        }
        
        workgroupBarrier();
    }
    
    // Write result
    let c_idx = i_global * params.n + j_global;
    c[c_idx] = params.alpha * sum + params.beta * c[c_idx];
}
```

---

## 4. Element-Wise Operations Template

### Example: ReLU
```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn relu(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    output[idx] = max(0.0, input[idx]);
}
```

### Example: Richards Curve
```wgsl
struct RichardsParams {
    nu: f32,
    k: f32,
    m: f32,
    beta: f32,
    temp_reciprocal: f32,
    output_gain: f32,
    output_bias: f32,
    scale: f32,
    shift: f32,
    // ... more params
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: RichardsParams;
@group(0) @binding(3) var<uniform> size: u32;

fn richards_curve(x: f32, p: RichardsParams) -> f32 {
    // CPU implementation: log(nu + k * exp(m * (x - beta)))
    let scaled_x = p.m * (x - p.beta);
    let exp_term = exp(clamp(scaled_x, -100.0, 100.0));  // Avoid overflow
    let numerator = p.nu + p.k * exp_term;
    return log(numerator + 1e-10);  // Avoid log(0)
}

@compute @workgroup_size(256)
fn richards_gate(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    output[idx] = richards_curve(input[idx], params);
}
```

---

## 5. Data Binding Pattern

### Bind Group Layout
```rust
let bind_group_layout = device.create_bind_group_layout(
    &wgpu::BindGroupLayoutDescriptor {
        label: Some("GEMM layout"),
        entries: &[
            // A matrix (storage buffer, read-only)
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
            // B matrix
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // C matrix (storage buffer, read-write)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    }
);
```

### Bind Group Creation
```rust
let bind_group = device.create_bind_group(
    &wgpu::BindGroupDescriptor {
        label: Some("GEMM bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    }
);
```

---

## 6. Rust Implementation Wrapper

```rust
impl WgpuMatrixOps {
    pub fn gemm_f32(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        alpha: f32,
        a: &GpuBuffer,
        b: &GpuBuffer,
        beta: f32,
        output: &mut GpuBuffer,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        // Create params struct
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct GemmParams {
            m: u32,
            n: u32,
            k: u32,
            alpha: f32,
            beta: f32,
            trans_a: u32,
            trans_b: u32,
            _pad: u32,
        }
        
        let params = GemmParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            alpha,
            beta,
            trans_a: trans_a as u32,
            trans_b: trans_b as u32,
            _pad: 0,
        };
        
        // Upload params to GPU
        let params_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("GEMM params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
        
        // Create bind group and execute
        let bind_group = self.device.create_bind_group(...);
        
        let mut encoder = self.device.create_command_encoder(...);
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GEMM compute pass"),
            });
            cpass.set_pipeline(&self.gemm_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch: (M + 15) / 16 groups in X, (N + 15) / 16 in Y
            let gx = ((m as u32) + 15) / 16;
            let gy = ((n as u32) + 15) / 16;
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        
        Ok(())
    }
}
```

---

## 7. Testing Pattern

### CPU Reference Test
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_gemm_f32_against_cpu() {
        let mut device = GpuDevice::auto_detect().expect("GPU required");
        
        // CPU reference
        let a = Array2::from_elem((128, 64), 2.0f32);
        let b = Array2::from_elem((64, 128), 3.0f32);
        let mut c_cpu = Array2::zeros((128, 128));
        ndarray_linalg::Gemm::gemm(2.0, &a, &b, 0.0, &mut c_cpu);
        
        // GPU computation
        let a_gpu = device.allocate_f32(128 * 64)?;
        let b_gpu = device.allocate_f32(64 * 128)?;
        let mut c_gpu = device.allocate_f32(128 * 128)?;
        
        device.upload(a.as_slice().unwrap(), &mut a_gpu_buf)?;
        device.upload(b.as_slice().unwrap(), &mut b_gpu_buf)?;
        
        device.gemm_f32(
            2.0, &a_gpu, &b_gpu, 0.0, &mut c_gpu,
            128, 128, 64, false, false
        )?;
        
        // Compare
        let mut c_result = vec![0.0f32; 128 * 128];
        device.download(&c_gpu, &mut c_result)?;
        
        for (gpu_val, cpu_val) in c_result.iter().zip(c_cpu.iter()) {
            let err = (gpu_val - cpu_val).abs();
            assert!(err <= 1e-4 * cpu_val.abs(), "Error {} > tolerance", err);
        }
    }
}
```

---

## 8. Performance Optimization Checklist

- [ ] Use workgroup_size(256) or (16, 16) to maximize occupancy
- [ ] Use shared memory (`var<workgroup>`) for frequent access patterns
- [ ] Coalesce global memory reads (consecutive threads read consecutive elements)
- [ ] Avoid branching in hot loops (all threads follow same path)
- [ ] Use barrier() between shared memory phases
- [ ] Profile with NVIDIA NSight or AMD uProf to identify bottlenecks
- [ ] Target: Memory bandwidth saturation for element-wise ops

---

## 9. Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Buffer overflow in shared memory | Check array bounds; use min() |
| Uninitialized shared memory | Call workgroupBarrier() after load |
| Race conditions | Use atomics or barrier() before shared reads |
| Numerical underflow/overflow | Clamp log() arguments, use exp bounds |
| Incorrect memory layout | Verify row-major (C) vs col-major (Fortran) |
| Pipeline not compiled | Check shader syntax; enable validation layers |

---

## 10. Shader Validation

### Enable WGPU Validation
```rust
let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
    backends: wgpu::Backends::all(),
    dx12_shader_compiler: Default::default(),
    flags: wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION,
});
```

### Shader Syntax Checker
```bash
# Use WGSL LSP server in VSCode
# Or validate offline: cargo install naga-cli
naga check shader.wgsl
```

---

## Summary

**Key takeaways**:
1. Each GPU operation needs: WGSL kernel + Rust wrapper + bind group setup
2. Use tiling/shared memory for BLAS (better memory reuse)
3. Use simple global-dispatch for element-wise ops
4. Always test against CPU reference (tolerance ≤ 1e-4)
5. Profile and optimize hot paths (GEMM, attention)

**Next step**: Implement GEMM first (Task 2A.1), then validate, then iterate on other ops.
