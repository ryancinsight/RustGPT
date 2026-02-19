# GPU Component Trait Implementation Plan - Phase 5.6

## Objective
Implement the `GpuComponent` trait for all shared components (SharedFeedforward, SharedAttentionContext, SharedTemporalProcessing) to provide unified GPU management.

## Current State

### What Exists
1. **GpuComponent Trait** (`src/domain/compute/gpu_component.rs`)
   - `set_gpu_device()`: Attach pre-configured GPU
   - `enable_gpu_auto_detect()`: Auto-detect GPU (strict no-fallback)
   - `is_gpu_ready()`: Check if GPU is ready
   - `gpu_backend_name()`: Get backend name
   - `gpu_device()`: Get the GPU device
   - `ensure_capacity()`: Pre-allocate buffers

2. **SharedFeedforward** (`src/domain/layers/components/feedforward.rs`)
   - Has `enable_gpu_auto_detect()` method
   - Uses ComputeBackend for dispatch
   - Does NOT implement GpuComponent trait

3. **SharedAttentionContext** (`src/domain/layers/components/attention_context.rs`)
   - CPU-only implementation
   - No GPU methods

4. **SharedTemporalProcessing** (`src/domain/layers/components/temporal_processing.rs`)
   - CPU-only implementation
   - Stubs in temporal_processing_gpu.rs

## Implementation Tasks

### Task 1: SharedFeedforward - GpuComponent Implementation

**File**: `src/domain/layers/components/feedforward.rs`

**Required Changes**:
1. Add GPU device field to struct:
   ```rust
   pub struct SharedFeedforward {
       feedforward: FeedForwardVariant,
       last_batch_size: Option<usize>,
       last_embed_dim: Option<usize>,
       compute_backend: ComputeBackend,
       gpu_device: Option<Arc<Mutex<GpuDevice>>>,  // NEW
   }
   ```

2. Implement GpuComponent trait:
   ```rust
   impl GpuComponent for SharedFeedforward {
       fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
           self.gpu_device = Some(device);
       }
       
       fn enable_gpu_auto_detect(&mut self) -> Result<()> {
           let device = GpuDevice::auto_detect()?;
           self.gpu_device = Some(Arc::new(Mutex::new(device)));
           Ok(())
       }
       
       fn is_gpu_ready(&self) -> bool {
           self.gpu_device.is_some()
       }
       
       fn gpu_backend_name(&self) -> Option<&'static str> {
           self.gpu_device.as_ref()
               .and_then(|d| d.lock().ok())
               .map(|guard| guard.backend().as_str())
       }
       
       fn gpu_device(&self) -> Option<Arc<Mutex<GpuDevice>>> {
           self.gpu_device.clone()
       }
       
       fn ensure_capacity(&mut self, batch_size: usize, embed_dim: usize, _seq_len: usize) -> Result<()> {
           // Pre-allocate GPU buffers for the given dimensions
           if let Some(device) = &self.gpu_device {
               let device_guard = device.lock().map_err(|_| ModelError::Backend {
                   message: "Failed to acquire GPU device lock".to_string(),
               })?;
               
               let ffn_size = batch_size * embed_dim;
               let intermediate_size = batch_size * (embed_dim * 4); // Typical hidden multiplier
               
               // Allocate input, intermediate, output buffers
               let _input = device_guard.allocate_f32(ffn_size)?;
               let _intermediate = device_guard.allocate_f32(intermediate_size)?;
               let _output = device_guard.allocate_f32(ffn_size)?;
           }
           Ok(())
       }
   }
   ```

**Dependencies**: 
- GpuDevice import
- Arc<Mutex> types

---

### Task 2: SharedAttentionContext - GpuComponent Implementation

**File**: `src/domain/layers/components/attention_context.rs`

**Current Status**: CPU-only, no GPU support

**Implementation Plan**:
1. Add GPU device and GPU context fields
2. Implement GpuComponent trait
3. Create `forward_gpu()` method for context modulation
4. Kernel needed: Context matrix fusion (batch norm + learned context)

**Kernel Pseudocode**:
```wgsl
// Input: x (batch_size x embed_dim)
// Context: C (embed_dim x embed_dim)
// Output: x @ C (batch_size x embed_dim)

fn attention_context_forward(
    x: &GpuBuffer,
    context_matrix: &GpuBuffer,
    output: &mut GpuBuffer,
    batch_size: usize,
    embed_dim: usize
) {
    // GEMM: output = x @ context_matrix
    // gamma = layernorm(output)
    // delta = apply_learned_delta(gamma)
    // return x * gamma + delta
}
```

---

### Task 3: SharedTemporalProcessing - GpuComponent Implementation

**File**: `src/domain/layers/components/temporal_processing.rs`

**Current Status**: CPU-only with placeholder GPU stubs

**Implementation Plan**:
1. Add GPU device field
2. Implement GpuComponent trait
3. Replace placeholder kernels in `temporal_processing_gpu.rs`:
   - PolyAttention: Polynomial basis computation + gating
   - Mamba/RG-LRU: Recurrent scan kernel
   - TransformerAttention: Scaled dot-product attention

**Kernels Needed**:
- `poly_attention_forward`: Q, K projections → polynomial scores → softmax → context
- `mamba_selective_scan`: State space model with input projection and selective gating
- `scaled_dot_product_attention`: Standard attention with numerical stability

---

## Integration with UnifiedGpuBufferPool

Once GpuComponent is implemented for all shared components, they can use the unified buffer pool:

```rust
// In each component's ensure_capacity method:
let pool = UnifiedGpuBufferPool::auto_detect()?;
pool.ensure_attention_buffers(batch_size, num_heads, seq_len, head_dim)?;

// Retrieve stats for monitoring
let stats = pool.allocation_stats();
println!("Memory efficiency: {:.1}%", stats.efficiency_percent());
println!("Reuse operations: {}", stats.reuse_count);
```

---

## Testing Strategy

### Unit Tests
1. GPU device attachment
2. Auto-detection error handling
3. Buffer pre-allocation correctness
4. Backend name reporting

### Integration Tests
1. End-to-end forward pass (GPU)
2. Numerical accuracy vs CPU reference
3. Multiple components sharing GPU device
4. Memory efficiency with stats

### Example Test
```rust
#[test]
fn test_shared_feedforward_gpu_component() {
    let mut component = SharedFeedforward::new(...);
    
    // Test auto-detection
    component.enable_gpu_auto_detect().expect("GPU should be available");
    assert!(component.is_gpu_ready());
    assert!(component.gpu_backend_name().is_some());
    
    // Test buffer pre-allocation
    component.ensure_capacity(32, 512, 64).expect("Capacity ensure failed");
    
    // Test forward pass
    let input = Array2::zeros((32, 512));
    let output = component.forward(&input);
    assert_eq!(output.dim(), (32, 512));
}
```

---

## Timeline

**Phase 5.6a** (Current):
- ✅ Remove CpuGpuMatrixOps deprecation
- ✅ Implement AllocationStats
- 🔄 Implement GpuComponent for SharedFeedforward
- 🔄 Implement GpuComponent for SharedAttentionContext
- 🔄 Implement GpuComponent for SharedTemporalProcessing

**Phase 5.6b**:
- Replace placeholder GPU kernels with WGPU implementations
- Add GPU kernel tests and benchmarks
- Performance profiling and optimization

**Phase 5.7+**:
- CUDA backend implementation
- Metal backend implementation
- Distributed GPU support

---

## Success Criteria

- ✅ All shared components implement GpuComponent trait
- ✅ All 539+ unit tests pass
- ✅ Zero compiler warnings
- ✅ GPU auto-detection works with strict no-fallback
- ✅ Memory efficiency ≥ 85% (from AllocationStats)
- ✅ Numerical accuracy within 1e-4 vs CPU reference
