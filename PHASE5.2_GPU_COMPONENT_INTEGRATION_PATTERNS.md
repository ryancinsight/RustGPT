# GPU Component Integration Patterns - Phase 5.2

**Purpose**: Reference guide for adding GPU paths to shared components  
**Status**: Ready for implementation  
**Target Components**: 7 high-impact shared components

---

## Pattern 1: Simple Element-Wise Operations

### Before (CPU-only)
```rust
pub struct SharedRichardsNorm {
    compute_backend: ComputeBackend,
}

impl SharedRichardsNorm {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        self.compute_backend.require_cpu_implemented("SharedRichardsNorm::forward");
        // CPU implementation
    }
}
```

### After (CPU + GPU)
```rust
pub struct SharedRichardsNorm {
    compute_backend: ComputeBackend,
    cached_gamma: Vec<f32>,
    cached_beta: Vec<f32>,
}

impl SharedRichardsNorm {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        self.compute_backend.require_cpu_implemented("SharedRichardsNorm::forward");
        // CPU implementation
    }
    
    pub fn forward_gpu(&self, device: &mut GpuDevice, x: &GpuBuffer) -> Result<GpuBuffer> {
        // Allocate output buffer
        let mut output = device.allocate_f32(x.size_f32())?;
        
        // Upload parameters if needed
        let gamma = device.allocate_f32(self.cached_gamma.len())?;
        device.ops.upload(&self.cached_gamma, &mut gamma)?;
        let beta = device.allocate_f32(self.cached_beta.len())?;
        device.ops.upload(&self.cached_beta, &mut beta)?;
        
        // Call LayerNorm kernel
        device.layer_norm(x, &gamma, &beta, &mut output, 
                          input.dim().0, input.dim().1, 1e-5)?;
        
        // Cleanup
        device.deallocate(gamma);
        device.deallocate(beta);
        
        Ok(output)
    }
}
```

**Key pattern**:
1. Allocate output buffer on device
2. Upload parameters if not already on device
3. Call GPU operation through device
4. Deallocate intermediate buffers
5. Return GPU buffer handle

---

## Pattern 2: Matrix Operations with Reuse

### SharedFeedforward.forward_into()

**GPU variant**:
```rust
pub fn forward_gpu_into(
    &mut self,
    device: &mut GpuDevice,
    input: &GpuBuffer,     // Batch size × embed_dim
    output: &mut GpuBuffer, // Pre-allocated output
) -> Result<()> {
    match &mut self.feedforward {
        FeedForwardVariant::RichardsGlu(glu) => {
            glu.forward_gpu_into(device, input, output)
        }
        FeedForwardVariant::MixtureOfExperts(moe) => {
            moe.forward_gpu_into(device, input, output)
        }
    }
}

// Inside RichardsGlu:
pub fn forward_gpu_into(
    &mut self,
    device: &mut GpuDevice,
    input: &GpuBuffer,
    output: &mut GpuBuffer,
) -> Result<()> {
    let (batch, embed_dim) = (input.size_f32() / embed_dim, embed_dim);
    
    // Allocate intermediate buffers
    let mut hidden1 = device.allocate_f32(batch * hidden_dim)?;
    let mut hidden2 = device.allocate_f32(batch * hidden_dim)?;
    
    // Projection: input @ W1 → hidden1
    device.gemm_f32(1.0, input, &self.w1_gpu, 0.0, &mut hidden1, 
                    batch, hidden_dim, embed_dim)?;
    
    // Projection: input @ W2 → hidden2  
    device.gemm_f32(1.0, input, &self.w2_gpu, 0.0, &mut hidden2,
                    batch, hidden_dim, embed_dim)?;
    
    // Apply activation to hidden1
    device.ops.gelu(input, &mut hidden1, hidden1.size_f32())?;
    
    // Element-wise multiply (gate): output = hidden1 * hidden2
    device.ops.axpy(1.0, &hidden1, 0.0, &hidden2, output, 
                    batch * hidden_dim)?;
    
    // Projection: hidden @ Wout → output
    device.gemm_f32(1.0, &output_partial, &self.wout_gpu, 0.0, output,
                    batch, embed_dim, hidden_dim)?;
    
    // Cleanup
    device.deallocate(hidden1);
    device.deallocate(hidden2);
    
    Ok(())
}
```

**Key pattern**:
- Allocate intermediate buffers for layer computations
- Chain operations via GEMM + element-wise ops
- Deallocate intermediates but reuse output buffer
- All operations on device (minimal CPU sync)

---

## Pattern 3: Attention with Caching

### SharedAttentionContext.update_outgoing_context_gpu()

```rust
pub fn update_outgoing_context_gpu(
    &mut self,
    device: &mut GpuDevice,
    temporal_out: &GpuBuffer,  // (batch, seq, embed_dim)
) -> Result<GpuBuffer> {
    let (batch, seq, embed) = self.context_shape;
    
    // Allocate similarity matrix (batch × embed)
    let mut similarity = device.allocate_f32(batch * embed)?;
    
    // Compute channel similarities via GEMM
    // temporal_out @ temporal_out^T
    device.gemm_f32(1.0, temporal_out, temporal_out, 0.0,
                    &mut similarity, batch, embed, seq)?;
    
    // Allocate context pooling output
    let mut pooled = device.allocate_f32(batch * embed)?;
    
    // Pooling: context = softmax(similarity) @ temporal_out
    let mut softmax_out = device.allocate_f32(batch * embed)?;
    device.softmax(&similarity, &mut softmax_out, batch, embed)?;
    
    device.gemm_f32(1.0, &softmax_out, temporal_out, 0.0,
                    &mut pooled, batch, embed, seq)?;
    
    // Cleanup temporaries
    device.deallocate(similarity);
    device.deallocate(softmax_out);
    
    Ok(pooled)
}
```

**Key pattern**:
- Use GEMM for similarity computation
- Use softmax for attention weights
- Use GEMM again for weighted sum
- Intermediate buffers deallocated after use

---

## Pattern 4: Recurrent Operations (Mamba/RG-LRU)

### Mamba.forward_gpu_with_state()

```rust
pub fn forward_gpu_with_state(
    &mut self,
    device: &mut GpuDevice,
    x: &GpuBuffer,              // (batch, embed_dim)
    state: &mut Option<GpuBuffer>, // Recurrent state (batch, state_dim)
) -> Result<GpuBuffer> {
    // Project input
    let mut proj = device.allocate_f32(x.size_f32() * proj_dim)?;
    device.gemm_f32(1.0, x, &self.input_proj_gpu, 0.0, &mut proj, 
                    batch, proj_dim, embed_dim)?;
    
    // Apply state transition (recurrence)
    if let Some(prev_state) = state {
        // h_new = A @ h_prev + B @ x_proj
        let mut next_state = device.allocate_f32(prev_state.size_f32())?;
        
        device.gemm_f32(1.0, &self.a_gpu, prev_state, 0.0, &mut next_state,
                        state_dim, state_dim, state_dim)?;
        device.gemm_f32(1.0, &self.b_gpu, &proj, 1.0, &mut next_state,
                        state_dim, proj_dim, proj_dim)?;
        
        // Update state
        *state = Some(next_state);
    } else {
        // Initialize state
        let mut next_state = device.allocate_f32(batch * state_dim)?;
        device.gemm_f32(1.0, &self.b_gpu, &proj, 0.0, &mut next_state,
                        state_dim, proj_dim, proj_dim)?;
        *state = Some(next_state);
    }
    
    // Output projection
    let mut output = device.allocate_f32(x.size_f32())?;
    let state_ref = state.as_ref().unwrap();
    device.gemm_f32(1.0, state_ref, &self.output_proj_gpu, 0.0, &mut output,
                    batch, embed_dim, state_dim)?;
    
    device.deallocate(proj);
    
    Ok(output)
}
```

**Key pattern**:
- Manage state buffer across calls
- Use GEMM for linear transformations
- Update state in-place
- Return output while preserving state

---

## Pattern 5: Diffusion-Specific Operations

### DiffusionBlock.forward_with_timestep_gpu()

```rust
pub fn forward_with_timestep_gpu(
    &mut self,
    device: &mut GpuDevice,
    x: &GpuBuffer,         // Input tensor
    timestep: u32,         // Diffusion step
) -> Result<GpuBuffer> {
    // Compute time embedding
    let time_emb = self.time_embedding.embed_gpu(device, timestep)?;
    
    // Process through temporal layer
    let mut norm1 = device.allocate_f32(x.size_f32())?;
    self.pre_attn_norm.forward_gpu(device, x, &mut norm1)?;
    
    let mut attn_out = device.allocate_f32(x.size_f32())?;
    self.temporal_mixing.forward_gpu(device, &norm1, &mut attn_out)?;
    
    // FiLM modulation with time embedding
    let mut modulated = device.allocate_f32(x.size_f32())?;
    self.film_modulation.apply_with_time_embedding_gpu(
        device, &attn_out, &time_emb, &mut modulated)?;
    
    // Residual
    let mut residual1 = device.allocate_f32(x.size_f32())?;
    device.add_scaled(1.0, x, &mut modulated.clone(), x.size_f32())?;
    
    // Continue with FFN...
    let mut norm2 = device.allocate_f32(x.size_f32())?;
    self.pre_ffn_norm.forward_gpu(device, &residual1, &mut norm2)?;
    
    let mut ffn_out = device.allocate_f32(x.size_f32())?;
    self.feedforward.forward_gpu_into(device, &norm2, &mut ffn_out)?;
    
    // Final residual
    let mut output = device.allocate_f32(x.size_f32())?;
    device.add_scaled(1.0, &residual1, &mut ffn_out, x.size_f32())?;
    
    // Cleanup
    device.deallocate(time_emb);
    device.deallocate(norm1);
    device.deallocate(attn_out);
    device.deallocate(modulated);
    device.deallocate(residual1);
    device.deallocate(norm2);
    device.deallocate(ffn_out);
    
    Ok(output)
}
```

**Key pattern**:
- Time embeddings computed on GPU
- FiLM modulation fused with activations
- Residual connections via add_scaled
- Cleanup intermediate buffers

---

## Pattern 6: Backward Pass Integration

### Adding Gradient Support

```rust
pub struct SharedTemporalProcessing {
    // ... forward state
    
    // Cache for backward
    #[serde(skip)]
    cached_input: Option<Arc<GpuBuffer>>,
    cached_temporal_output: Option<Arc<GpuBuffer>>,
}

impl SharedTemporalProcessing {
    pub fn forward_gpu_with_cache(
        &mut self,
        device: &mut GpuDevice,
        x: &GpuBuffer,
    ) -> Result<GpuBuffer> {
        // Forward pass
        let output = self.forward_gpu(device, x)?;
        
        // Cache for backward
        self.cached_input = Some(Arc::new(x.clone()));
        self.cached_temporal_output = Some(Arc::new(output.clone()));
        
        Ok(output)
    }
    
    pub fn backward_gpu(
        &self,
        device: &mut GpuDevice,
        output_grads: &GpuBuffer,
    ) -> Result<(GpuBuffer, Vec<GpuBuffer>)> {
        let input = self.cached_input.as_ref().ok_or_else(|| {
            ModelError::Backend { 
                message: "No cached input for backward pass".to_string() 
            }
        })?;
        
        // Gradient computation via GPU kernels
        // ...
        
        Ok((input_grads, param_grads))
    }
}
```

**Key pattern**:
- Cache input/output tensors as Arc<GpuBuffer>
- Use cached values in backward pass
- Deallocate caches after training step

---

## Pattern 7: Memory Optimization Checklist

For each GPU component implementation:

- [ ] **Pre-allocate buffers**: Use UnifiedLayerWorkspace sizes
- [ ] **Reuse across calls**: Keep persistent buffers (parameters)
- [ ] **Cleanup intermediates**: Deallocate within function scope
- [ ] **Track memory**: Call device.memory_stats() for debugging
- [ ] **Batch operations**: Combine small GEMMs into larger operations
- [ ] **Fuse kernels**: Combine norm + activation + residual (if backend supports)
- [ ] **Minimize transfers**: Keep data on GPU during full forward pass
- [ ] **Test numerics**: Compare vs CPU output with ε ≤ 1e-4

---

## Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| Memory leaks (buffers not deallocated) | Use RAII pattern: allocate at scope entry, deallocate at scope exit |
| Silent errors (GPU operation fails, silent NaN) | Always check Result<>, log errors |
| CPU-GPU synchronization too frequent | Batch operations, minimize CPU access during forward pass |
| Numerical divergence (GPU != CPU) | Use ε ≤ 1e-4 tolerance in tests, check float precision |
| Wrong dimensions in GEMM | Verify (m, n, k) match matrix dimensions before call |
| State not preserved across calls | Use Arc<Mutex<>> for shared mutable state |

---

## Testing Each Pattern

### Unit Test Template
```rust
#[cfg(feature = "gpu-cuda")]
#[test]
fn temporal_processing_gpu_vs_cpu() {
    let mut component = SharedTemporalProcessing::new(/* ... */);
    component.set_compute_backend(ComputeBackend::Cpu);
    
    let x_cpu = /* random input */;
    let cpu_out = component.forward(&x_cpu);
    
    component.set_compute_backend(ComputeBackend::Cuda);
    let mut device = GpuDevice::new(ComputeBackend::Cuda).unwrap();
    let x_gpu = device.allocate_f32(x_cpu.size()).unwrap();
    device.upload(&x_cpu.raw_data(), &mut x_gpu).unwrap();
    
    let gpu_out_buf = component.forward_gpu(&mut device, &x_gpu).unwrap();
    let mut gpu_out_cpu = vec![0.0; gpu_out_buf.size_f32()];
    device.download(&gpu_out_buf, &mut gpu_out_cpu).unwrap();
    let gpu_out = Array2::from_shape_vec(shape, gpu_out_cpu).unwrap();
    
    assert_abs_diff_eq!(cpu_out, gpu_out, epsilon = 1e-4);
}
```

---

## Next Steps Checklist

- [ ] Choose first component to implement (recommend: RichardsNorm)
- [ ] Implement CUDA backend with GEMM support
- [ ] Apply Pattern 1 (element-wise ops) to RichardsNorm
- [ ] Write unit test following template
- [ ] Move to Pattern 2 (feedforward) after GEMM validated
- [ ] Benchmark each component
- [ ] Document observed speedups and memory usage

---

## References

- **Pattern source**: `src/domain/compute/{gpu_device,gpu_ops}.rs`
- **Component target**: `src/domain/layers/components/*.rs`
- **Test utilities**: `src/domain/compute/gpu_device.rs` (memory tracking)
- **Integration guide**: `GPU_BACKEND_IMPLEMENTATION_STRATEGY.md`
