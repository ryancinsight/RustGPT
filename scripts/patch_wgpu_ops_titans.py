"""Append Titans dispatch implementations to wgpu_ops.rs"""

# Parameter structs and dispatch implementations
WGPU_IMPL = '''
// ============================================================================
// Titans Memory kernel dispatch (WgpuMatrixOps impl)
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TitansMlpParams {
    num_tokens: u32,
    key_dim: u32,
    hidden_dim: u32,
    val_dim: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TitansGradW2Params {
    num_tokens: u32,
    hidden_dim: u32,
    val_dim: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TitansGradW1Params {
    num_tokens: u32,
    key_dim: u32,
    hidden_dim: u32,
    val_dim: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TitansUpdateParams {
    num_elements: u32,
    alpha: f32,
    eta: f32,
    theta: f32,
}

impl GpuMatrixOps for WgpuMatrixOps {
    // This block provides the Titans-specific override implementations added
    // on top of the existing `impl GpuMatrixOps for WgpuMatrixOps` block.
    // Rust allows multiple impl blocks for the same type/trait so these
    // declarations live in a separate impl block for clarity.
}

// NOTE: Titans kernel dispatch methods are added directly on `WgpuMatrixOps`
// (not via the trait) so they can be called from NeuralMemory::forward_gpu_kernel
// via the concrete type. They mirror the trait signatures.

impl WgpuMatrixOps {
    /// Dispatches SHADER_TITANS_MLP_FORWARD.
    pub fn dispatch_titans_mlp_forward(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        keys: &GpuBuffer,
        w1: &GpuBuffer,
        b1: &GpuBuffer,
        w2: &GpuBuffer,
        b2: &GpuBuffer,
        z_out: &mut GpuBuffer,
        h_out: &mut GpuBuffer,
        v_pred: &mut GpuBuffer,
        num_tokens: usize,
        key_dim: usize,
        hidden_dim: usize,
        val_dim: usize,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        use wgpu::CommandEncoderDescriptor;

        let params = TitansMlpParams {
            num_tokens: num_tokens as u32,
            key_dim: key_dim as u32,
            hidden_dim: hidden_dim as u32,
            val_dim: val_dim as u32,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Titans MLP Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buf_keys   = Self::resolve_buffer(pool, keys.id)?;
        let buf_w1     = Self::resolve_buffer(pool, w1.id)?;
        let buf_b1     = Self::resolve_buffer(pool, b1.id)?;
        let buf_w2     = Self::resolve_buffer(pool, w2.id)?;
        let buf_b2     = Self::resolve_buffer(pool, b2.id)?;
        let buf_z_out  = Self::resolve_buffer(pool, z_out.id)?;
        let buf_h_out  = Self::resolve_buffer(pool, h_out.id)?;
        let buf_vpred  = Self::resolve_buffer(pool, v_pred.id)?;

        let pipeline = self.get_or_create_pipeline(
            "titans_mlp_forward",
            crate::domain::compute::wgsl_kernels::SHADER_TITANS_MLP_FORWARD,
            &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Titans MLP Bind Group"),
            layout: &self.bind_group_layouts["titans_mlp_forward"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_keys.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_w1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_b1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_w2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_b2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf_z_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: buf_h_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: buf_vpred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("Titans MLP Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Titans MLP Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (num_tokens as u32 + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Dispatches SHADER_TITANS_GRAD_W2.
    pub fn dispatch_titans_grad_w2(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        v_target: &GpuBuffer,
        v_pred: &GpuBuffer,
        h_act: &GpuBuffer,
        grad_w2: &mut GpuBuffer,
        grad_b2: &mut GpuBuffer,
        num_tokens: usize,
        hidden_dim: usize,
        val_dim: usize,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        use wgpu::CommandEncoderDescriptor;

        let params = TitansGradW2Params {
            num_tokens: num_tokens as u32,
            hidden_dim: hidden_dim as u32,
            val_dim: val_dim as u32,
            pad: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Titans GradW2 Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buf_vtgt  = Self::resolve_buffer(pool, v_target.id)?;
        let buf_vpred = Self::resolve_buffer(pool, v_pred.id)?;
        let buf_hact  = Self::resolve_buffer(pool, h_act.id)?;
        let buf_gw2   = Self::resolve_buffer(pool, grad_w2.id)?;
        let buf_gb2   = Self::resolve_buffer(pool, grad_b2.id)?;

        let pipeline = self.get_or_create_pipeline(
            "titans_grad_w2",
            crate::domain::compute::wgsl_kernels::SHADER_TITANS_GRAD_W2,
            &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Titans GradW2 Bind Group"),
            layout: &self.bind_group_layouts["titans_grad_w2"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_vtgt.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_vpred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_hact.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_gw2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_gb2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("Titans GradW2 Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Titans GradW2 Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // 2D dispatch: (val_dim, hidden_dim+1) workgroups of (1,1,1)
            cpass.dispatch_workgroups(val_dim as u32, (hidden_dim as u32) + 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Dispatches SHADER_TITANS_GRAD_W1.
    pub fn dispatch_titans_grad_w1(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        keys: &GpuBuffer,
        v_target: &GpuBuffer,
        v_pred: &GpuBuffer,
        z: &GpuBuffer,
        w2: &GpuBuffer,
        grad_w1: &mut GpuBuffer,
        grad_b1: &mut GpuBuffer,
        num_tokens: usize,
        key_dim: usize,
        hidden_dim: usize,
        val_dim: usize,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        use wgpu::CommandEncoderDescriptor;

        let params = TitansGradW1Params {
            num_tokens: num_tokens as u32,
            key_dim: key_dim as u32,
            hidden_dim: hidden_dim as u32,
            val_dim: val_dim as u32,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Titans GradW1 Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buf_keys  = Self::resolve_buffer(pool, keys.id)?;
        let buf_vtgt  = Self::resolve_buffer(pool, v_target.id)?;
        let buf_vpred = Self::resolve_buffer(pool, v_pred.id)?;
        let buf_z     = Self::resolve_buffer(pool, z.id)?;
        let buf_w2    = Self::resolve_buffer(pool, w2.id)?;
        let buf_gw1   = Self::resolve_buffer(pool, grad_w1.id)?;
        let buf_gb1   = Self::resolve_buffer(pool, grad_b1.id)?;

        let pipeline = self.get_or_create_pipeline(
            "titans_grad_w1",
            crate::domain::compute::wgsl_kernels::SHADER_TITANS_GRAD_W1,
            &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Titans GradW1 Bind Group"),
            layout: &self.bind_group_layouts["titans_grad_w1"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_keys.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_vtgt.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_vpred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_w2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf_gw1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: buf_gb1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("Titans GradW1 Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Titans GradW1 Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // 2D dispatch: (hidden_dim, key_dim+1) workgroups of (1,1,1)
            cpass.dispatch_workgroups(hidden_dim as u32, (key_dim as u32) + 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Dispatches SHADER_TITANS_MEMORY_UPDATE.
    pub fn dispatch_titans_memory_update(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        grad: &GpuBuffer,
        momentum: &mut GpuBuffer,
        memory: &mut GpuBuffer,
        num_elements: usize,
        alpha: f32,
        eta: f32,
        theta: f32,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        use wgpu::CommandEncoderDescriptor;

        let params = TitansUpdateParams {
            num_elements: num_elements as u32,
            alpha,
            eta,
            theta,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Titans Update Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buf_grad = Self::resolve_buffer(pool, grad.id)?;
        let buf_mom  = Self::resolve_buffer(pool, momentum.id)?;
        let buf_mem  = Self::resolve_buffer(pool, memory.id)?;

        let pipeline = self.get_or_create_pipeline(
            "titans_memory_update",
            crate::domain::compute::wgsl_kernels::SHADER_TITANS_MEMORY_UPDATE,
            &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Titans Update Bind Group"),
            layout: &self.bind_group_layouts["titans_memory_update"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_grad.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_mom.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_mem.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("Titans Update Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Titans Update Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (num_elements as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}
'''

# We need to insert these implementations BEFORE the fallback comment block
target = 'd:/RustGPT/src/domain/compute/wgpu_ops.rs'
content = open(target, 'r', encoding='utf-8').read()
marker = '// ============================================================================\n// Non-wgpu fallback (stub implementations)\n// ============================================================================'
if marker not in content:
    print('ERROR: marker not found in wgpu_ops.rs')
    exit(1)

# Remove the final empty `impl GpuMatrixOps for WgpuMatrixOps {}` that we added
BOGUS_IMPL = '''
impl GpuMatrixOps for WgpuMatrixOps {
    // This block provides the Titans-specific override implementations added
    // on top of the existing `impl GpuMatrixOps for WgpuMatrixOps` block.
    // Rust allows multiple impl blocks for the same type/trait so these
    // declarations live in a separate impl block for clarity.
}
'''
impl_block = WGPU_IMPL.replace(BOGUS_IMPL, '')

new_content = content.replace(marker, impl_block + '\n' + marker, 1)
open(target, 'w', encoding='utf-8').write(new_content)
print(f'wgpu_ops.rs: inserted Titans dispatch implementations ({len(impl_block)} bytes)')
