# GPU Kernel Consolidation & Optimization - Phase 5.6.3 Execution Plan

**Date**: Feb 16, 2026  
**Thread**: @T-019c6753-5d92-72de-b050-d422c54bfd65  
**Focus**: Consolidate shared components (Diffusion, SSM, Transformer) with GPU backend variants, automatic detection, strict no-fallback

## Execution Roadmap

### Phase 3A: Unified GPU Kernel Cleanup & Documentation
**Status**: IN PROGRESS

#### Tasks
1. **Fix diagnostics** ✓
   - Remove unused imports from `unified_gpu_backend.rs` (Array1)
   - Clean up GPU shared executor imports

2. **Implement GPU Buffer Management**
   - [ ] Add `GpuBuffer` type alias definition
   - [ ] Add `GpuMemoryPool` trait implementation
   - [ ] Implement buffer lifecycle tracking in `GpuKernelWorkspace`

3. **Consolidate Kernel Implementations**
   - [ ] **RichardsGLU Fused Kernel**: Two-pass strategy
     - Pass 1: W1 projection → Richards activation → (batch, hidden_dim)
     - Pass 2: W2 projection → Gate → Output
     - Use workspace buffers to avoid intermediate allocations
   
   - [ ] **Selective Scan (Mamba)**: GPU kernel dispatcher
   - [ ] **RG-LRU Recurrent**: GPU kernel dispatcher
   - [ ] **Attention Operations**: QKV projection, softmax, V projection

### Phase 3B: Automatic GPU Detection Implementation
**Status**: TODO

#### Tasks
1. **Strict GPU Auto-Detection**
   - [ ] Implement `GpuDevice::auto_detect()` with priority: CUDA > Metal > Vulkan
   - [ ] Ensure NO CPU fallback - return error if no GPU available
   - [ ] Add detailed error messages for missing GPU features

2. **Error Handling & Troubleshooting**
   - [ ] Implement `ModelError::Backend` for GPU operation failures
   - [ ] Add context to error messages (which operation failed, why)
   - [ ] Create troubleshooting guide for common GPU issues

### Phase 3C: Memory Efficiency Optimization
**Status**: TODO

#### Tasks
1. **Power-of-2 Buffer Sizing**
   - [ ] Align buffers to 256-byte boundaries
   - [ ] Implement `GpuKernelWorkspace::ensure_capacity()` with power-of-2 rounding
   - [ ] Track reuse count to measure buffer efficiency

2. **Zero-Copy Pipeline**
   - [ ] Keep data on GPU between operations
   - [ ] Implement streaming buffers for attention computation
   - [ ] Minimize upload/download cycles

3. **Workspace Statistics**
   - [ ] Track buffer allocations and reallocations
   - [ ] Monitor bytes uploaded/downloaded
   - [ ] Measure kernel launch overhead

### Phase 3D: Performance Testing & Verification
**Status**: TODO

#### Target Performance Metrics
- **RichardsGLU**: 25x speedup (50ms → 2ms on 1K batch)
- **Multi-head Attention**: 30x speedup (30ms → 1ms on 512 batch)
- **Mamba Selective Scan**: 20x speedup (40ms → 2ms on 512 batch)
- **RG-LRU Recurrent**: 15x speedup (30ms → 2ms on 512 batch)

#### Tests to Implement
1. [ ] Microbenchmarks for each kernel type
2. [ ] Integration tests with actual model forward passes
3. [ ] Memory profiling under load
4. [ ] GPU vs CPU performance comparison

## Architecture Decisions

### Kernel Dispatch Strategy
```
UnifiedGpuKernels
├── auto_detect() → GpuDevice (strict no-fallback)
├── activation_forward() → GpuActivation enum dispatch
├── attention_forward() → AttentionParams-based dispatch
├── ssm_forward() → SsmParams-based dispatch
└── workspace management → GpuKernelWorkspace
    └── power-of-2 buffer sizing
    └── reuse tracking
    └── statistics collection
```

### Memory Management
- **Workspace-based**: Pre-allocate buffers at launch
- **No deallocations**: Buffers persist until `cleanup_workspace()`
- **Power-of-2 sizing**: For efficient GPU memory coalescing
- **Reuse metrics**: Track buffer reuse vs reallocation

### Error Handling
- **No CPU fallback**: If GPU operation fails, return `ModelError::Backend`
- **Detailed messages**: Include operation name, backend type, failure reason
- **Strict requirements**: Forces developers to implement all GPU kernels

## Implementation Order

1. **[NOW]** Fix diagnostics and imports
2. **[NEXT]** Implement `GpuBuffer` type and memory management
3. **Implement RichardsGLU fused kernel** (highest priority)
4. **Implement attention kernels** (critical path)
5. **Implement SSM kernels** (Mamba, RG-LRU)
6. **Testing and performance validation**
7. **Documentation and examples**

## Code Patterns

### Creating a GPU Kernel
```rust
// 1. Define parameters
pub struct MyKernelParams {
    // ... parameters
}

// 2. Implement kernel dispatch in UnifiedGpuKernels
pub fn my_kernel_forward(
    &mut self,
    input: &Array2<f32>,
    params: &MyKernelParams,
) -> Result<Array2<f32>> {
    // 3. Get GPU device
    let mut device = self.device.lock().map_err(|_| ModelError::Backend {
        message: "Failed to acquire GPU device lock".to_string(),
    })?;
    
    // 4. Ensure workspace capacity
    self.ensure_capacity(params.batch_size, params.embed_dim, params.seq_len)?;
    
    // 5. Allocate/reuse buffers
    let mut input_buf = device.allocate(input_size)?;
    
    // 6. Upload data
    device.upload(input.as_slice().unwrap(), &mut input_buf)?;
    
    // 7. Execute kernel
    device.my_kernel_op(&input_buf, &mut output_buf)?;
    
    // 8. Download result
    let mut output = vec![0.0f32; output_size];
    device.download(&output_buf, &mut output)?;
    
    // 9. Return reshaped output
    Ok(Array2::from_shape_vec((batch_size, dim), output)?)
}
```

## Success Criteria

- [x] Compilation passes without errors
- [ ] GPU auto-detection works (CUDA > Metal > Vulkan priority)
- [ ] RichardsGLU kernel achieves 25x speedup target
- [ ] Attention kernels achieve 30x speedup target
- [ ] All GPU operations have zero CPU fallback
- [ ] Memory efficiency improved (buffer reuse, power-of-2 sizing)
- [ ] Comprehensive error messages for GPU failures
- [ ] Documentation and examples complete

