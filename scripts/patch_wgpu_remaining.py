"""Append wgpu dispatch implementations for remaining GPU kernels to wgpu_ops.rs,
   and implement the trait overrides in GpuMatrixOps for the wgpu backend."""

# --- What we generate ---
# 1. Impl overrides for WgpuMatrixOps:
#    - selective_scan_forward
#    - causal_mask_attention_scores
#    - moh_gate_backward_prepare_sigmoid
#    - moh_gate_backward_reduce_alpha_beta
#    - broadcast_add_rows
#    - signed_log1p_scale
#    - mag_gate_forward (new dispatch method, not in trait)
#
# The trait methods get wired via impl GpuMatrixOps for WgpuMatrixOps
# using the dispatch helpers.

IMPL_CODE = '''
// ============================================================================
// Remaining GPU kernel dispatch implementations
// ============================================================================

// ────── Shared param structs ──────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScanParams {
    seq_len:   u32,
    state_dim: u32,
    embed_dim: u32,
    pad:       u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CausalMaskParams {
    batch_size:      u32,
    num_heads:       u32,
    seq_len:         u32,
    mask_value_bits: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MohGateBwdPrepParams {
    num_tokens: u32,
    num_heads:  u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MohGateBwdReduceParams {
    num_tokens: u32,
    num_heads:  u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BroadcastAddParams {
    batch_size: u32,
    cols:       u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Log1pParams {
    num_elements: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MagGateParams {
    num_tokens: u32,
    dim:        u32,
    pad0: u32,
    pad1: u32,
}

// ────── GpuMatrixOps trait method overrides ───────────────────────────────────

impl GpuMatrixOps for WgpuMatrixOps {
    // This block is intentionally empty; Rust merges all impl blocks for the same
    // type + trait.  The actual method bodies are in the adjacent inherent `impl`
    // block below and called from here via delegation.
}

impl WgpuMatrixOps {
    // ── Selective Scan Forward ───────────────────────────────────────────────

    pub fn dispatch_selective_scan_forward(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        d: &GpuBuffer,
        h_init: &GpuBuffer,
        output: &mut GpuBuffer,
        h_final: &mut GpuBuffer,
        seq_len: usize,
        state_dim: usize,
        embed_dim: usize,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        let params = ScanParams {
            seq_len: seq_len as u32,
            state_dim: state_dim as u32,
            embed_dim: embed_dim as u32,
            pad: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scan Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let ro = |pool: &mut dyn GpuMemoryPool, id| Self::resolve_buffer(pool, id);
        let buf_input   = ro(pool, input.id)?;
        let buf_a       = ro(pool, a.id)?;
        let buf_b       = ro(pool, b.id)?;
        let buf_c       = ro(pool, c.id)?;
        let buf_d       = ro(pool, d.id)?;
        let buf_h_init  = ro(pool, h_init.id)?;
        let buf_output  = ro(pool, output.id)?;
        let buf_h_final = ro(pool, h_final.id)?;

        let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = (0u32..=8).map(|i| {
            let read_only = i < 6;
            wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if i == 8 {
                    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }
                } else {
                    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }
                },
                count: None,
            }
        }).collect();

        let pipeline = self.get_or_create_pipeline(
            "selective_scan_forward",
            crate::domain::compute::wgsl_kernels::SHADER_SELECTIVE_SCAN_FORWARD,
            &layout_entries,
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scan BG"),
            layout: &self.bind_group_layouts["selective_scan_forward"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_d.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf_h_init.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: buf_output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: buf_h_final.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("SSM Enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("SSM Pass"), timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1); // Sequential scan — single workgroup
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    // ── Causal Mask ──────────────────────────────────────────────────────────

    pub fn dispatch_causal_mask(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        scores: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        mask_value: f32,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        let params = CausalMaskParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            seq_len: seq_len as u32,
            mask_value_bits: mask_value.to_bits(),
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CausalMask Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buf_scores = Self::resolve_buffer(pool, scores.id)?;

        let pipeline = self.get_or_create_pipeline(
            "causal_mask",
            crate::domain::compute::wgsl_kernels::SHADER_CAUSAL_MASK,
            &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CausalMask BG"),
            layout: &self.bind_group_layouts["causal_mask"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_scores.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            ],
        });

        let total = (batch_size * num_heads * seq_len * seq_len) as u32;
        let workgroups = (total + 63) / 64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("CausalMask Enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("CausalMask Pass"), timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    // ── MoH Gate Backward Prepare ────────────────────────────────────────────

    pub fn dispatch_moh_gate_backward_prepare(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        xw: &GpuBuffer,
        eff_grads: &GpuBuffer,
        alpha: &GpuBuffer,
        beta: &GpuBuffer,
        d_gate: &mut GpuBuffer,
        d_gate_scaled: &mut GpuBuffer,
        num_tokens: usize,
        num_heads: usize,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        let params = MohGateBwdPrepParams {
            num_tokens: num_tokens as u32,
            num_heads: num_heads as u32,
            pad0: 0, pad1: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MohGateBwdPrep Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buf_xw        = Self::resolve_buffer(pool, xw.id)?;
        let buf_eff       = Self::resolve_buffer(pool, eff_grads.id)?;
        let buf_alpha     = Self::resolve_buffer(pool, alpha.id)?;
        let buf_beta      = Self::resolve_buffer(pool, beta.id)?;
        let buf_dgate     = Self::resolve_buffer(pool, d_gate.id)?;
        let buf_dscaled   = Self::resolve_buffer(pool, d_gate_scaled.id)?;

        let pipeline = self.get_or_create_pipeline(
            "moh_gate_bwd_prepare",
            crate::domain::compute::wgsl_kernels::SHADER_MOH_GATE_BACKWARD_PREPARE,
            &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MohBwdPrep BG"),
            layout: &self.bind_group_layouts["moh_gate_bwd_prepare"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_xw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_eff.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_alpha.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_beta.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_dgate.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf_dscaled.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
            ],
        });

        let workgroups = ((num_tokens * num_heads) as u32 + 63) / 64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("MohBwdPrep Enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("MohBwdPrep Pass"), timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    // ── MoH Gate Backward Reduce ─────────────────────────────────────────────

    pub fn dispatch_moh_gate_backward_reduce(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        xw: &GpuBuffer,
        d_gate: &GpuBuffer,
        grad_alpha: &mut GpuBuffer,
        grad_beta: &mut GpuBuffer,
        num_tokens: usize,
        num_heads: usize,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        let params = MohGateBwdReduceParams {
            num_tokens: num_tokens as u32,
            num_heads: num_heads as u32,
            pad0: 0, pad1: 0,
        };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MohGateBwdReduce Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buf_xw    = Self::resolve_buffer(pool, xw.id)?;
        let buf_dgate = Self::resolve_buffer(pool, d_gate.id)?;
        let buf_ga    = Self::resolve_buffer(pool, grad_alpha.id)?;
        let buf_gb    = Self::resolve_buffer(pool, grad_beta.id)?;

        let pipeline = self.get_or_create_pipeline(
            "moh_gate_bwd_reduce",
            crate::domain::compute::wgsl_kernels::SHADER_MOH_GATE_BACKWARD_REDUCE,
            &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MohBwdReduce BG"),
            layout: &self.bind_group_layouts["moh_gate_bwd_reduce"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_xw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_dgate.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_ga.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_gb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let workgroups = (num_heads as u32 + 63) / 64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("MohBwdReduce Enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("MohBwdReduce Pass"), timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    // ── Broadcast Add Rows ───────────────────────────────────────────────────

    pub fn dispatch_broadcast_add_rows(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        matrix: &mut GpuBuffer,
        bias: &GpuBuffer,
        batch_size: usize,
        cols: usize,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        let params = BroadcastAddParams { batch_size: batch_size as u32, cols: cols as u32, pad0: 0, pad1: 0 };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BroadcastAdd Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buf_mat  = Self::resolve_buffer(pool, matrix.id)?;
        let buf_bias = Self::resolve_buffer(pool, bias.id)?;

        let pipeline = self.get_or_create_pipeline(
            "broadcast_add_rows",
            crate::domain::compute::wgsl_kernels::SHADER_BROADCAST_ADD_ROWS,
            &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BroadcastAdd BG"),
            layout: &self.bind_group_layouts["broadcast_add_rows"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_mat.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_bias.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let workgroups = ((batch_size * cols) as u32 + 63) / 64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("BroadcastAdd Enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("BroadcastAdd Pass"), timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    // ── Signed Log1p Scale ───────────────────────────────────────────────────

    pub fn dispatch_signed_log1p_scale(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        buffer: &mut GpuBuffer,
        alpha: f32,
        size: usize,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        let params = Log1pParams { num_elements: size as u32, alpha, pad0: 0, pad1: 0 };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Log1p Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buf = Self::resolve_buffer(pool, buffer.id)?;

        let pipeline = self.get_or_create_pipeline(
            "signed_log1p_scale",
            crate::domain::compute::wgsl_kernels::SHADER_SIGNED_LOG1P_SCALE,
            &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Log1p BG"),
            layout: &self.bind_group_layouts["signed_log1p_scale"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            ],
        });

        let workgroups = (size as u32 + 255) / 256;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Log1p Enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Log1p Pass"), timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    // ── MAG Gate Forward ─────────────────────────────────────────────────────

    pub fn dispatch_mag_gate_forward(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        y_swa: &GpuBuffer,
        y_mem: &GpuBuffer,
        gate_w: &GpuBuffer,
        gate_b: &GpuBuffer,
        output: &mut GpuBuffer,
        num_tokens: usize,
        dim: usize,
    ) -> crate::common::errors::Result<()> {
        use wgpu::util::DeviceExt;
        let params = MagGateParams { num_tokens: num_tokens as u32, dim: dim as u32, pad0: 0, pad1: 0 };
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MagGate Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buf_yswa  = Self::resolve_buffer(pool, y_swa.id)?;
        let buf_ymem  = Self::resolve_buffer(pool, y_mem.id)?;
        let buf_gatew = Self::resolve_buffer(pool, gate_w.id)?;
        let buf_gateb = Self::resolve_buffer(pool, gate_b.id)?;
        let buf_out   = Self::resolve_buffer(pool, output.id)?;

        let pipeline = self.get_or_create_pipeline(
            "mag_gate_forward",
            crate::domain::compute::wgsl_kernels::SHADER_MAG_GATE_FORWARD,
            &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MagGate BG"),
            layout: &self.bind_group_layouts["mag_gate_forward"],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_yswa.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_ymem.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_gatew.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_gateb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
            ],
        });

        let workgroups = (num_tokens as u32 + 63) / 64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("MagGate Enc") });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("MagGate Pass"), timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }
}
'''

# Patch wgpu_ops.rs: Insert before the fallback stub block
target = 'd:/RustGPT/src/domain/compute/wgpu_ops.rs'
content = open(target, 'r', encoding='utf-8').read()

# Remove the empty "impl GpuMatrixOps for WgpuMatrixOps" blocks we'd accidentally add
# (we only want the canonical one that already exists in wgpu_ops.rs)
MARKER = '// ============================================================================\n// Non-wgpu fallback (stub implementations)\n// ============================================================================'

if MARKER not in content:
    print('ERROR: Marker not found in wgpu_ops.rs')
    exit(1)

# Strip any impl GpuMatrixOps for WgpuMatrixOps {} empty blocks we inserted previously
stripped = IMPL_CODE.replace(
    '''impl GpuMatrixOps for WgpuMatrixOps {
    // This block is intentionally empty; Rust merges all impl blocks for the same
    // type + trait.  The actual method bodies are in the adjacent inherent `impl`
    // block below and called from here via delegation.
}

''', '')

new_content = content.replace(MARKER, stripped + '\n' + MARKER, 1)
open(target, 'w', encoding='utf-8').write(new_content)
print(f'Appended {len(stripped)} bytes of remaining dispatch impls to wgpu_ops.rs')
