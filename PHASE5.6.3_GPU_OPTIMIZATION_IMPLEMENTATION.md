# Phase 5.6.3: GPU Optimization Implementation Guide

**Date**: February 15, 2026  
**Objective**: Complete GPU backend integration and optimize shared components with strict no-fallback semantics

---

## I. Current Status Assessment

### ✅ Completed Infrastructure
1. **GPU Device Abstraction** (`gpu_device.rs`)
   - ✅ `GpuDevice::auto_detect()` with no CPU fallback
   - ✅ Priority: CUDA > Metal > Vulkan
   - ✅ Clear error messages when GPU unavailable
   - ✅ Conditional feature compilation guards

2. **GPU Component Trait** (`gpu_component.rs`)
   - ✅ Unified trait for all shared components
   - ✅ Strict error semantics for no GPU
   - ✅ Device attachment and auto-detection methods
   - ✅ Capacity pre-allocation for zero-allocation forward

3. **Shared Components with Full GpuComponent Support**
   - ✅ **SharedAttentionContext**: Lines 1085-1132
     - Implements all trait methods
     - Pre-allocates input, activation, context, similarity, softmax buffers
   - ✅ **SharedFeedforward**: Lines 473-533
     - Implements all trait methods
     - Tracks batch/embed dimensions for FFN sizing
     - Delegates GPU device to underlying feedforward variant
   - ✅ **SharedTemporalProcessing**: Lines 717-771
     - Implements all trait methods
     - Allocates temporal, attention score buffers
     - Delegates GPU device to underlying temporal mixing layer

### ⚠️ In Progress
1. **RichardsGLU Fused Kernel** (`richards_glu_fused_kernel.rs`)
   - ✅ CPU reference implementation complete
   - ✅ Parameter structure defined (`OptimizedRichardsGluParams`)
   - ✅ Intermediate buffer management (`RichardsGluIntermediates`)
   - ⏳ GPU kernel dispatch (CUDA/Metal/WGPU) - PENDING
   - ⏳ GPU forward implementation - PENDING

2. **Unified GPU Executor** (`unified_gpu_executor.rs`)
   - ✅ Struct definition and basic methods
   - ✅ SharedAttentionContext kernel stubs
   - ✅ SharedFeedforward kernel stubs
   - ✅ SharedTemporalProcessing kernel stubs
   - ⏳ Actual kernel implementations - PENDING

---

## II. Work Priorities (High → Low)

### Priority 1: Complete RichardsGLU GPU Integration
**Why**: Feedforward is bottleneck; fused kernel reduces 5→2 GPU launches

**Tasks**:
1. Implement GPU kernel dispatch in `UnifiedGpuExecutor`
2. Wire RichardsGLU fused kernel to `SharedFeedforward::forward_gpu()`
3. Benchmark 5-launch vs 2-pass performance
4. Add integration test with gradient validation

**Files to Modify**:
- `src/domain/compute/unified_gpu_executor.rs` (kernel dispatch)
- `src/domain/compute/richards_glu_fused_kernel.rs` (GPU launch code)
- `src/domain/layers/components/feedforward.rs` (GPU forward wire)

---

### Priority 2: Consolidate Shared Component GPU Paths
**Why**: Ensure zero-copy data flow through all layers

**Tasks**:
1. Verify `SharedAttentionContext::apply_context_gpu()` calls GPU path first
2. Verify `SharedFeedforward::forward_gpu()` uses unified executor
3. Verify `SharedTemporalProcessing::forward_gpu()` dispatches to temporal variant
4. Test zero-copy forward: input → attention → temporal → feedforward → output

**Files to Verify**:
- `src/domain/layers/components/attention_context.rs` (apply_context methods)
- `src/domain/layers/components/feedforward.rs` (forward_gpu method)
- `src/domain/layers/components/temporal_processing.rs` (forward_gpu method)

---

### Priority 3: PolyAttention GPU Optimization
**Why**: Attention is compute bottleneck; fused kernel reduces 3+→1 GPU launches

**Tasks**:
1. Implement fused Q/K/V projection + scoring + softmax kernel
2. Add GPU support to `PolyAttention` struct
3. Implement `GpuComponent` for PolyAttention (if not already done)
4. Benchmark poly attention: 512 batch, 768 dims

**Files to Modify**:
- `src/domain/attention/poly_attention.rs` (GPU forward)
- `src/domain/compute/unified_gpu_executor.rs` (PolyAttention kernel dispatch)

---

### Priority 4: Cleanup & Verification
**Why**: Ensure correctness and remove dead code paths

**Tasks**:
1. Remove duplicate CPU-only implementations (if any)
2. Audit all `#[cfg(...)]` guards for proper conditional compilation
3. Verify all tests pass with GPU and CPU-only builds
4. Update benchmark suite to measure GPU vs CPU

**Commands**:
```bash
# CPU-only build
cargo build --release
cargo test --lib

# All GPU backends
cargo build --release --features gpu-all
cargo test --lib
cargo test --test gpu_shared_components_phase56

# Verify no silent fallback
cargo test --lib -- --nocapture | grep -i "no gpu"
```

---

## III. Implementation Details

### A. RichardsGLU Fused Kernel Completion

**Current State**: CPU reference + parameter structure ready  
**Next Steps**: GPU kernel dispatch

#### Step 1: Extend `OptimizedRichardsGluParams` for GPU

```rust
// In richards_glu_fused_kernel.rs

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct RichardsGluGpuConfig {
    pub params: OptimizedRichardsGluParams,
    pub use_streaming: bool,  // Keep data on GPU between passes
    pub benchmark_mode: bool, // Skip unnecessary copies
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn forward_gpu(
    device: &mut GpuDevice,
    input: &GpuBuffer,
    w1: &GpuBuffer,
    w2: &GpuBuffer,
    w_out: &GpuBuffer,
    config: &RichardsGluGpuConfig,
) -> Result<GpuBuffer> {
    // Pass 1: x1 = input @ w1, x2 = input @ w2
    //         value = x1 * richards(x1), gate = richards(x2)
    //         gated = value * gate
    
    // Pass 2: output = gated @ w_out
    
    // Return output buffer
}
```

#### Step 2: Wire to `UnifiedGpuExecutor`

```rust
// In unified_gpu_executor.rs

pub fn forward_richards_glu(
    &mut self,
    input: &GpuBuffer,
    w1: &GpuBuffer,
    w2: &GpuBuffer,
    w_out: &GpuBuffer,
    params: &OptimizedRichardsGluParams,
) -> Result<GpuBuffer> {
    richards_glu_fused_kernel::forward_gpu(
        self.device_mut(),
        input,
        w1,
        w2,
        w_out,
        &RichardsGluGpuConfig {
            params: *params,
            use_streaming: true,
            benchmark_mode: false,
        },
    )
}
```

#### Step 3: Wire to `SharedFeedforward`

```rust
// In feedforward.rs - SharedFeedforward::forward_gpu()

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn forward_gpu(
    &mut self,
    input: &GpuBuffer,
    device: &mut GpuDevice,
) -> Result<GpuBuffer> {
    match &mut self.feedforward {
        FeedforwardKind::RichardsGlu { 
            w1, w2, w_out, params, .. 
        } => {
            // Get GPU buffers for weights
            let w1_gpu = /* allocate and upload w1 */;
            let w2_gpu = /* allocate and upload w2 */;
            let w_out_gpu = /* allocate and upload w_out */;
            
            // Dispatch to unified executor
            UnifiedGpuExecutor::new_with_device(device)?
                .forward_richards_glu(
                    input,
                    &w1_gpu,
                    &w2_gpu,
                    &w_out_gpu,
                    params,
                )
        }
        FeedforwardKind::MixtureOfExperts { .. } => {
            Err(ModelError::NotImplemented {
                feature: "GPU MoE forward".to_string(),
            })
        }
    }
}
```

---

### B. Shared Component GPU Wiring Verification

#### Checklist for Each Component

**SharedAttentionContext**:
- [ ] Method `apply_incoming_context_gpu()` exists and calls GPU path
- [ ] Fallback to CPU only if GPU is not available (currently should error)
- [ ] GPU context modulation correct: `output = input + (strength/embed_dim) * (input @ context)`
- [ ] GPU similarity computation correct: `similarity = activation @ activation^T` then softmax

**SharedFeedforward**:
- [ ] Method `forward_gpu()` exists and dispatches to RichardsGLU GPU kernel
- [ ] Zero-allocation forward (no intermediate CPU arrays)
- [ ] Batch processing without device sync

**SharedTemporalProcessing**:
- [ ] Dispatcher method `forward_gpu()` routes based on `TemporalMixingType`
- [ ] Attention variant uses attention GPU kernels
- [ ] Mamba variant uses selective scan GPU kernel
- [ ] RG-LRU variant uses RG-LRU GPU kernel

---

### C. PolyAttention GPU Optimization Strategy

**Current**: CPU-based polynomial attention  
**Target**: Fused GPU kernel

#### Fused Pipeline (1 GPU Launch)
```
Input: Q, K, V (batch_size, seq_len, embed_dim)

Kernel (single pass):
  1. Q @ W_q, K @ W_k, V @ W_v  (projections)
  2. Scores = Q @ K^T / sqrt(d_k)  (attention scores)
  3. Softmax(scores, dim=-1)  (attention weights)
  4. Output = Attention @ V  (context aggregation)
  
Output: (batch_size, seq_len, embed_dim)
```

**vs Current** (3+ passes):
- Pass 1: Projections
- Pass 2: Scoring
- Pass 3: Softmax
- Pass 4: Context aggregation

---

## IV. Testing Strategy

### Unit Tests
1. **GPU Device Auto-Detection**
   ```rust
   #[test]
   fn test_gpu_device_no_fallback() {
       match GpuDevice::auto_detect() {
           Ok(device) => {
               assert!(device.backend().is_gpu());
               println!("GPU: {}", device.name());
           }
           Err(e) => {
               // Expected if no GPU available
               assert!(e.to_string().contains("no supported GPU"));
           }
       }
   }
   ```

2. **Shared Component GPU Readiness**
   ```rust
   #[test]
   fn test_shared_components_gpu_ready() {
       let mut context = SharedAttentionContext::new(...);
       let mut feedforward = SharedFeedforward::new(...);
       let mut temporal = SharedTemporalProcessing::new(...);
       
       // Should error before enable_gpu_auto_detect()
       assert!(!context.is_gpu_ready());
       assert!(!feedforward.is_gpu_ready());
       assert!(!temporal.is_gpu_ready());
       
       // Try auto-detect
       match context.enable_gpu_auto_detect() {
           Ok(()) => {
               assert!(context.is_gpu_ready());
               // All components share same device
               feedforward.set_gpu_device(context.gpu_device().unwrap());
               temporal.set_gpu_device(context.gpu_device().unwrap());
               assert!(feedforward.is_gpu_ready());
               assert!(temporal.is_gpu_ready());
           }
           Err(e) => {
               println!("No GPU available: {}", e);
           }
       }
   }
   ```

3. **RichardsGLU CPU vs GPU Validation**
   ```rust
   #[test]
   fn test_richards_glu_gpu_accuracy() {
       let batch_size = 8;
       let input_dim = 768;
       let hidden_dim = 3072;
       let output_dim = 768;
       
       // CPU reference
       let cpu_output = forward_reference_cpu(...);
       
       // GPU computation (if available)
       if let Ok(mut device) = GpuDevice::auto_detect() {
           let gpu_output = forward_gpu(&mut device, ...)?;
           
           // Download and compare
           let mut gpu_result = vec![0.0; cpu_output.len()];
           device.download(&gpu_output, &mut gpu_result)?;
           
           // Verify numerical accuracy (small epsilon)
           assert_arrays_close(&cpu_output.data(), &gpu_result, 1e-5);
       }
   }
   ```

### Integration Tests
1. **Zero-Copy Forward Pass** (`tests/gpu_shared_components_phase56.rs`)
   ```rust
   #[test]
   fn test_zero_copy_forward_pipeline() {
       // Input → AttentionContext → Temporal → Feedforward → Output
       // Verify no data downloaded between GPU operations
   }
   ```

2. **Gradient Flow with GPU** (if applicable)
   ```rust
   #[test]
   fn test_gpu_gradients_flow() {
       // Train small model on GPU
       // Verify gradients backprop through all components
   }
   ```

### Benchmark Tests
1. **RichardsGLU: 5-Launch vs 2-Pass**
   ```bash
   cargo bench --bench richards_glu_fused -- --nocapture
   ```
   Expected:
   - 5-launch naive: 50ms (1K batch)
   - 2-pass fused: 2ms (1K batch)
   - Speedup: 25x

2. **Shared Components Benchmark**
   ```bash
   cargo bench --bench shared_components_gpu -- --nocapture
   ```

---

## V. Consolidation Checklist

- [ ] **Phase 1: RichardsGLU GPU**
  - [ ] GPU kernel dispatch implemented
  - [ ] Unified executor wired
  - [ ] SharedFeedforward GPU path complete
  - [ ] Benchmark shows 25x speedup

- [ ] **Phase 2: Shared Component Consolidation**
  - [ ] AttentionContext GPU path verified
  - [ ] Feedforward GPU path verified
  - [ ] TemporalProcessing GPU path verified
  - [ ] Zero-copy forward test passes

- [ ] **Phase 3: PolyAttention GPU**
  - [ ] Fused kernel implemented
  - [ ] GpuComponent trait implemented
  - [ ] Benchmark shows 30x speedup

- [ ] **Phase 4: Cleanup & Verification**
  - [ ] No dead code paths
  - [ ] All conditional compilation works
  - [ ] All tests pass (GPU + CPU-only)
  - [ ] No silent CPU fallback

---

## VI. Success Criteria

1. **Performance**
   - RichardsGLU: 25x speedup on GPU (50ms → 2ms)
   - PolyAttention: 30x speedup on GPU (30ms → 1ms)
   - Mamba: 20x speedup on GPU (40ms → 2ms)

2. **Correctness**
   - GPU output matches CPU reference (numerical error < 1e-5)
   - Gradients flow correctly through GPU components

3. **No-Fallback Semantics**
   - `GpuDevice::auto_detect()` returns error if no GPU
   - `component.enable_gpu_auto_detect()` errors if no GPU
   - `component.forward_gpu()` errors if GPU not attached
   - NO silent CPU fallback anywhere

4. **Code Quality**
   - Zero unused code paths
   - Clear error messages for GPU failures
   - Proper feature guard compilation

---

## VII. Documentation Updates Needed

- [ ] Update `AGENTS.md` with GPU build commands
- [ ] Add GPU troubleshooting guide
- [ ] Document strict no-fallback semantics
- [ ] Add GPU optimization best practices
- [ ] Performance tuning guide for different batch sizes
