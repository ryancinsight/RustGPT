# Phase 5.4: GPU Forward Implementation Guide

**Status**: Ready for Implementation  
**Priority**: P1 - Critical for GPU acceleration  
**Estimated Time**: 3-4 hours per block type

---

## Overview

This guide details the specific GPU forward implementations needed to complete Phase 5.4:
1. **DiffusionBlock GPU variant**
2. **Mamba/RG-LRU recurrent kernels** (currently placeholders)
3. **TransformerBlock GPU end-to-end path** (verify completeness)

---

## 1. DiffusionBlock GPU Variant Implementation

### Current Status
- DiffusionBlock CPU path: Fully implemented in `src/domain/blocks/diffusion_block.rs`
- GPU variant: **Not implemented**
- Workspace: Uses `UnifiedLayerWorkspace`

### Implementation Plan

#### Step 1: Create `diffusion_block_gpu.rs`
```rust
// src/domain/blocks/diffusion_block_gpu.rs

use crate::common::errors::Result;
use crate::domain::compute::{GpuDevice, GpuComponent};
use ndarray::Array2;
use std::sync::{Arc, Mutex};

impl DiffusionBlock {
    /// GPU-accelerated forward pass
    ///
    /// Computes: output = diffusion_step(input, timestep, condition)
    ///
    /// GPU kernels used:
    /// 1. GEMM for projection layers
    /// 2. Softmax for attention (if applicable)
    /// 3. LayerNorm for normalization
    /// 4. Custom diffusion kernel for time/condition conditioning
    pub fn forward_gpu(&mut self, 
        input: &Array2<f32>, 
        timestep: f32,
        condition: Option<&Array2<f32>>
    ) -> Result<Array2<f32>> {
        // Step 1: Validate GPU is ready
        self.require_gpu_ready()?;
        
        // Step 2: Ensure GPU buffers
        let (batch_size, embed_dim) = (input.nrows(), input.ncols());
        self.gpu_workspace.ensure_capacity(batch_size, embed_dim, 1)?;
        
        // Step 3: Execute GPU kernels in order
        // 3a. Upload input to GPU
        let mut gpu_input = self.gpu_workspace.allocate_input(batch_size, embed_dim)?;
        self.gpu_device.upload(input.as_slice().unwrap(), &mut gpu_input)?;
        
        // 3b. Apply diffusion step (projection + timestep embedding + condition)
        let mut gpu_hidden = self.gpu_workspace.allocate_hidden(batch_size, self.hidden_dim)?;
        self.apply_diffusion_projection_gpu(&gpu_input, timestep, condition, &mut gpu_hidden)?;
        
        // 3c. Apply temporal mixing (self-attention or SSM depending on variant)
        let mut gpu_temporal = self.gpu_workspace.allocate_temporal(batch_size, embed_dim)?;
        self.temporal_processing.forward_gpu_variant(&gpu_hidden, &mut gpu_temporal)?;
        
        // 3d. Apply feedforward
        let mut gpu_ffn = self.gpu_workspace.allocate_ffn(batch_size, embed_dim)?;
        self.feedforward.forward_gpu(&gpu_temporal, &mut gpu_ffn)?;
        
        // 3e. Download result back to CPU
        let mut output = Array2::zeros((batch_size, embed_dim));
        self.gpu_device.download(&gpu_ffn, output.as_slice_mut().unwrap())?;
        
        Ok(output)
    }
    
    /// Apply diffusion-specific projection with timestep embedding
    fn apply_diffusion_projection_gpu(
        &mut self,
        input: &GpuBuffer,
        timestep: f32,
        condition: Option<&Array2<f32>>,
        output: &mut GpuBuffer,
    ) -> Result<()> {
        // GPU kernel: Project input + embed timestep + fuse condition if provided
        // Equivalent to: output = W @ input + timestep_embed + (condition if provided)
        
        // Implementation depends on actual projection matrices
        // For now, placeholder: output = input * scale + timestep_bias
        
        let scale = 1.0 / (1.0 + timestep.exp());
        self.gpu_device.scale(scale, output, output.size_bytes)?;
        
        if condition.is_some() {
            // Add condition weighting here
        }
        
        Ok(())
    }
}
```

#### Step 2: Implement `require_gpu_ready()` helper
```rust
fn require_gpu_ready(&self) -> Result<()> {
    if !self.is_gpu_ready {
        return Err(ModelError::Backend {
            message: "DiffusionBlock GPU requested without GPU attached. \
                     Call enable_gpu_auto_detect() first.".to_string(),
        });
    }
    Ok(())
}
```

#### Step 3: Add GPU field to DiffusionBlock struct
```rust
pub struct DiffusionBlock {
    // ... existing fields ...
    
    // GPU support (Phase 5.4)
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,
    gpu_workspace: UnifiedGpuBufferPool,
    is_gpu_ready: bool,
}
```

#### Step 4: Implement GpuComponent trait
```rust
impl GpuComponent for DiffusionBlock {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.gpu_device = Some(device);
    }
    
    fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;
        self.gpu_device = Some(Arc::new(Mutex::new(device)));
        self.is_gpu_ready = true;
        Ok(())
    }
    
    fn is_gpu_ready(&self) -> bool {
        self.is_gpu_ready && self.gpu_device.is_some()
    }
    
    fn gpu_backend_name(&self) -> Option<&'static str> {
        self.gpu_device.as_ref()
            .and_then(|d| d.lock().ok())
            .map(|guard| guard.backend().as_str())
    }
    
    fn ensure_capacity(&mut self, batch_size: usize, embed_dim: usize, seq_len: usize) -> Result<()> {
        // Ensure GPU buffers are sized for the given batch/embedding/sequence
        self.gpu_workspace.ensure_attention_context_buffers(batch_size, embed_dim)
    }
}
```

---

## 2. Mamba/RG-LRU Recurrent Kernel Completion

### Current Status
File: `src/domain/temporal/temporal_processing_gpu.rs`
- Lines 47-59: `forward_gpu_mamba()` - **Placeholder**
- Lines 95-119: `forward_gpu_rg_lru()` - **Placeholder**
- **Problem**: Both return `Ok(input.to_owned())` (passthrough)

### Implementation Plan

#### Step 1: Implement Mamba Recurrent Scan GPU Kernel

**CPU Reference**: `src/domain/temporal/mamba.rs`
- Selective scan: `S_t = A_t * S_{t-1} + B_t * X_t`
- Parametric: Uses learned A, B, C matrices that depend on input

**GPU Implementation**:
```rust
fn forward_gpu_mamba(
    &self,
    input: &Array2<f32>,
    ctx: &mut GpuSharedOpsContext,
    ops: &mut dyn GpuMatrixOps,
) -> Result<Array2<f32>> {
    let (batch_size, seq_len) = (input.nrows(), input.ncols());
    let hidden_dim = self.hidden_dim;
    
    // Step 1: Project input to A, B, C, X representations
    // X: input projection (seq_len, input_dim) -> (batch, seq_len, hidden_dim)
    let gpu_x = ctx.allocate_or_reuse(batch_size * seq_len * hidden_dim)?;
    ops.gemm_f32(1.0, &gpu_input, &self.gpu_weights.w_x, 0.0, &mut gpu_x, 
                 batch_size * seq_len, hidden_dim, input.ncols())?;
    
    // A: Learned parameter (usually fixed or slightly input-dependent)
    // B, C: Projected from input (seq_len, input_dim) -> (batch, seq_len, ssm_rank)
    
    // Step 2: Apply selective scan (recurrent operation)
    // h_t = A * h_{t-1} + B * x_t
    // y_t = C @ h_t
    let gpu_state = ctx.allocate_or_reuse(batch_size * hidden_dim)?;
    
    for t in 0..seq_len {
        // Apply selective scan step
        self.mamba_selective_scan_step_gpu(
            t, &gpu_x, &gpu_b, &gpu_c, 
            &mut gpu_state, &mut gpu_output
        )?;
    }
    
    // Step 3: Project back to output dimension
    let mut gpu_output = ctx.allocate_or_reuse(batch_size * seq_len * output_dim)?;
    ops.gemm_f32(1.0, &gpu_state, &self.gpu_weights.w_out, 0.0, &mut gpu_output,
                 batch_size * seq_len, output_dim, hidden_dim)?;
    
    // Step 4: Download to CPU
    let mut output = Array2::zeros((batch_size, seq_len));
    ops.download(&gpu_output, output.as_slice_mut().unwrap())?;
    
    Ok(output)
}

fn mamba_selective_scan_step_gpu(
    &self,
    t: usize,
    gpu_x: &GpuBuffer,
    gpu_b: &GpuBuffer,
    gpu_c: &GpuBuffer,
    gpu_state: &mut GpuBuffer,
    gpu_output: &mut GpuBuffer,
) -> Result<()> {
    // GPU kernel: Selective scan step
    // h = A * h + B[t] * x[t]
    // y = C[t] @ h
    
    // This requires a custom WGSL kernel or composition of existing ops
    // For now, pseudocode:
    
    // 1. Scale state by A (diagonal matrix multiplication)
    self.ops.scale(self.a_decay, gpu_state, gpu_state.size_bytes)?;
    
    // 2. Add input contribution: state += B[t] * x[t]
    self.ops.axpy(1.0, gpu_b_t, 1.0, gpu_x_t, gpu_state, gpu_state.size_bytes)?;
    
    // 3. Project to output: output[t] = C[t] @ state
    // This is a GEMV operation (matrix-vector multiply)
    self.ops.gemv_f32(1.0, gpu_c, gpu_state, 0.0, gpu_output, ...)?;
    
    Ok(())
}
```

**WGSL Shader** (new file: `src/domain/compute/wgpu_ops.rs` - add to shader pipeline):
```wgsl
// Mamba selective scan step shader
@compute @workgroup_size(256)
fn mamba_selective_scan_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    
    // Load state, A, B, x, C from buffers
    let h_prev = stateBuffer[idx];
    let a_val = aBuffer[idx];
    let b_val = bBuffer[idx];
    let x_val = xBuffer[idx];
    let c_val = cBuffer[idx];
    
    // h_t = a * h_{t-1} + b * x_t
    let h_new = a_val * h_prev + b_val * x_val;
    
    // y_t = c * h_t
    let y_val = c_val * h_new;
    
    // Write back
    stateBuffer[idx] = h_new;
    outputBuffer[idx] = y_val;
}
```

#### Step 2: Implement RG-LRU Recurrent Kernel

**CPU Reference**: `src/domain/temporal/rg_lru.rs`
- Recurrence: `h_t = λ * h_{t-1} + (1-λ) * x_t` with learnable λ
- Output: `y_t = gates(h_t) * (W @ h_t)`

**GPU Implementation** (similar pattern to Mamba):
```rust
fn forward_gpu_rg_lru(
    &self,
    input: &Array2<f32>,
    ctx: &mut GpuSharedOpsContext,
    ops: &mut dyn GpuMatrixOps,
) -> Result<Array2<f32>> {
    let (batch_size, seq_len) = (input.nrows(), input.ncols());
    
    // Step 1: Project input
    let gpu_x = ctx.allocate_or_reuse(...)?;
    ops.gemm_f32(...)?;  // Project to hidden dimension
    
    // Step 2: Apply RG-LRU recurrence
    let gpu_state = ctx.allocate_or_reuse(...)?;
    for t in 0..seq_len {
        self.rg_lru_recurrence_step_gpu(t, &gpu_x, &mut gpu_state, ...)?;
    }
    
    // Step 3: Apply output projection with gating
    let gpu_output = ctx.allocate_or_reuse(...)?;
    ops.gemm_f32(...)?;  // Output projection
    
    // Download and return
    ...
}

fn rg_lru_recurrence_step_gpu(
    &self,
    t: usize,
    gpu_x: &GpuBuffer,
    gpu_state: &mut GpuBuffer,
) -> Result<()> {
    // h_t = λ * h_{t-1} + (1-λ) * x_t
    // With λ learnable or computed from input
    
    let lambda = self.compute_lambda_gpu(gpu_x, t)?;
    let one_minus_lambda = 1.0 - lambda;
    
    // AXPY: h_t = λ * h_{t-1} + (1-λ) * x_t
    self.ops.axpy(
        lambda, 
        gpu_state, 
        one_minus_lambda, 
        gpu_x_t, 
        gpu_state, 
        gpu_state.size_bytes
    )?;
    
    Ok(())
}
```

---

## 3. TransformerBlock GPU End-to-End Verification

### Current Status
File: `src/domain/blocks/transformer_block.rs`
- Likely has GPU implementation started
- **Action**: Verify completeness of `forward_gpu()` path

### Verification Checklist
```rust
impl TransformerBlock {
    pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // [ ] Validate GPU ready (is_gpu_ready check)
        // [ ] Allocate GPU buffers for batch/sequence/embedding dims
        // [ ] Upload input to GPU
        
        // Attention path
        // [ ] Apply attention context on GPU (gpu_apply_attention_context)
        // [ ] Apply temporal mixing (attention/SSM variant) on GPU
        
        // Feedforward path
        // [ ] Apply feedforward on GPU (including RichardsGLU or MoE)
        // [ ] Apply activation functions on GPU
        
        // Output
        // [ ] Apply residual connections (in-place if possible)
        // [ ] Download output to CPU
        // [ ] Return output
    }
}
```

### Expected GPU Operations Sequence
1. **Attention Block**:
   - QKV projection (GEMM)
   - Softmax (custom kernel)
   - Output projection (GEMM)
   - Residual add (AXPY)

2. **Temporal Mixing** (PolyAttention variant):
   - Polynomial scoring (custom kernel)
   - Gating (Richards activation)
   - Output projection
   - Residual add

3. **Feedforward**:
   - First projection (GEMM)
   - Richards GLU gating (custom kernel)
   - Second projection (GEMM)
   - Residual add

---

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_diffusion_block_gpu_forward() {
    let mut block = DiffusionBlock::new(...);
    if let Err(_) = block.enable_gpu_auto_detect() {
        println!("No GPU available, skipping GPU test");
        return;
    }
    
    let input = Array2::random((2, 256));
    let output_gpu = block.forward_gpu(&input, 0.5, None).unwrap();
    
    // Verify output shape
    assert_eq!(output_gpu.shape(), input.shape());
}

#[test]
fn test_mamba_gpu_vs_cpu() {
    let mamba = Mamba::new(...);
    let input = Array2::random((4, 128));
    
    let output_cpu = mamba.forward_cpu(&input).unwrap();
    
    if let Ok(output_gpu) = mamba.forward_gpu(&input) {
        // Compare within numerical tolerance
        let max_diff = (output_cpu - output_gpu).mapv(f32::abs).max();
        assert!(max_diff < 1e-4, "GPU output differs from CPU");
    }
}
```

### Benchmark Tests
```rust
#[bench]
fn bench_transformer_gpu_forward(b: &mut Bencher) {
    let mut block = TransformerBlock::new(...);
    block.enable_gpu_auto_detect().ok();
    
    let input = Array2::random((32, 256));
    b.iter(|| block.forward_gpu(&input))
}
```

---

## Integration with Consolidation

These GPU implementations will use the consolidated APIs:
- **Device Management**: `UnifiedGpuBufferPool` (Phase 5.4.1)
- **Component Interface**: `GpuComponent` trait
- **Memory Management**: Unified buffer pooling with power-of-2 sizing
- **Strict GPU Mode**: No fallback to CPU on `forward_gpu()`

---

## Success Criteria

✅ DiffusionBlock has full `forward_gpu()` implementation  
✅ Mamba/RG-LRU placeholders replaced with actual recurrent kernels  
✅ TransformerBlock GPU path verified end-to-end  
✅ All GPU operations return `Result` (explicit error handling)  
✅ Numerical accuracy within 1e-4 vs CPU reference  
✅ 20-30% performance improvement measurable in benchmarks  
✅ 529+ tests passing with GPU consolidation APIs  

---

## References

- GPU Backend Architecture: GPU_BACKEND_IMPLEMENTATION_STATUS.md
- Current GPU Implementation: src/domain/compute/wgpu_ops.rs
- Consolidation Plan: SESSION_CONSOLIDATION_GPU_PHASE5.4_PLAN.md
