# GPU Consolidation Diagnostic Report
**Date**: Feb 16, 2026  
**Session**: Phase 5.6 - GPU Backend Consolidation

## Current State Assessment

### ✅ Completed (Phase 5.5)
1. **Unified GPU Kernels Infrastructure**
   - `unified_gpu_kernels.rs`: Dispatcher with workspace management
   - `unified_gpu_backend.rs`: Backend abstraction layer
   - `GpuComponent` trait: Standard interface for components

2. **GPU Memory Management**
   - `GpuKernelWorkspace`: Power-of-2 sizing, zero-copy reuse
   - Buffer pooling: Allocation count tracking
   - Memory statistics: upload/download bytes tracking

3. **GPU Device Abstraction**
   - `GpuDevice`: Core GPU context (CUDA/Metal/WGPU)
   - `GpuMatrixOps`: GEMM, softmax, activation operations
   - Feature flags: `gpu-cuda`, `gpu-metal`, `gpu-wgpu`

### ⚠️ Partially Complete (70-80%)
1. **SharedAttentionContext GPU** (`attention_context_gpu.rs`)
   - ✓ File exists with stubs
   - ✗ GPU forward pass incomplete
   - ✗ No numerical validation tests
   - ✗ Not integrated into attention_context.rs

2. **SharedFeedforward GPU** (`feedforward_gpu.rs`)
   - ✓ Basic structure defined
   - ✗ RichardsGLU fused kernel incomplete
   - ✗ Bias and activation still on CPU post-download
   - ✗ Workspace integration incomplete

3. **SharedTemporalProcessing GPU** (`temporal_processing_gpu.rs`)
   - ✓ Skeleton exists
   - ✗ Attention kernel not implemented
   - ✗ Mamba selective scan kernel not implemented
   - ✗ RG-LRU recurrent kernel not implemented

### ❌ Not Started
1. **GPU Backend Consolidation**
   - CUDA/Metal/WGPU auto-detection paths still duplicated
   - Memory pool implementations duplicated across backends
   - No unified error handling (CPU fallback still exists in some paths)

2. **Numerical Validation Tests**
   - No GPU vs CPU comparison tests
   - No tolerance verification (should be < 1e-4)
   - No cross-backend consistency tests

3. **Performance Benchmarks**
   - No latency measurements
   - No speedup verification (target: 25x-30x)
   - No memory usage profiling

---

## Code Quality Issues Found

### Issue 1: GPU Device Lock Patterns
**Problem**: Inconsistent lock handling across components

```rust
// ❌ Pattern 1: Unwrap (panics on lock failure)
let mut device = self.device.lock().unwrap();

// ⚠️ Pattern 2: map_err with generic message
let mut device = self.device.lock().map_err(|_| ModelError::Backend {
    message: "Failed to acquire GPU device lock".to_string(),
})?;

// ✓ Pattern 3: Consistent error context
let mut device = self.device.lock().map_err(|_| ModelError::Backend {
    message: "GPU device lock failed for [OPERATION_NAME]".to_string(),
})?;
```

**Action**: Standardize on Pattern 3 with context-specific error messages.

---

### Issue 2: Memory Management Inconsistency
**Problem**: Some code uses workspace pooling, others allocate per-operation

```rust
// ❌ No pooling (allocates new buffers each call)
let input_buf = device.allocate(input_size)?;
let output_buf = device.allocate(output_size)?;
// ... operation ...
device.deallocate(input_buf);
device.deallocate(output_buf);

// ✓ With pooling (reuses pre-allocated buffers)
let workspace = self.workspace.as_mut()?;
workspace.ensure_capacity(&mut device, batch, embed_dim)?;
// ... operation using workspace.buf_input, workspace.buf_output ...
workspace.reset();  // No deallocation - buffers reused
```

**Action**: Audit all GPU operations and convert to workspace pooling.

---

### Issue 3: Missing Strict No-Fallback Semantics
**Problem**: Some GPU code still falls back to CPU

```rust
// ❌ Anti-pattern: Silent fallback
pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    match self.forward_gpu(input) {
        Ok(output) => output,
        Err(_) => self.forward_cpu(input),  // Silent fallback!
    }
}

// ✓ Correct pattern: Fail fast
pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    if self.gpu_device.is_some() {
        return self.forward_gpu(input);  // Errors propagate
    }
    Ok(self.forward_cpu(input))
}
```

**Action**: Remove all CPU fallbacks from GPU paths. GPU operations should error if backend unavailable.

---

### Issue 4: Incomplete Fused Kernel Implementations
**Problem**: RichardsGLU kernel not fully fused (operations still separate)

```rust
// ❌ Current: 3 separate operations (3 kernel launches)
device.gemm_f32(&input, &w1, &mut hidden)?;        // Launch 1
device.richards_activation(&hidden)?;               // Launch 2
device.gemm_f32(&hidden, &w2, &mut output)?;       // Launch 3

// ✓ Target: Single fused kernel (1 launch)
device.richards_glu_fused(&input, &w1, &w2, &mut output)?;
```

**Action**: Implement true fused kernels in each backend (CUDA, Metal, WGPU).

---

## Missing Implementations (Detailed)

### 1. GpuDevice Auto-Detection
**Current State**:
- File: `src/domain/compute/gpu_device.rs`
- Issue: Code paths duplicated across backends
- Missing: Consolidated priority order (CUDA > Metal > Vulkan > WGPU)

**Implementation Needed**:
```rust
impl GpuDevice {
    pub fn auto_detect() -> Result<Self> {
        // Try CUDA first
        if cfg!(feature = "gpu-cuda") {
            if let Ok(device) = Self::new(ComputeBackend::Cuda) {
                return Ok(device);
            }
        }
        
        // Try Metal
        if cfg!(feature = "gpu-metal") {
            if let Ok(device) = Self::new(ComputeBackend::Metal) {
                return Ok(device);
            }
        }
        
        // Try Vulkan
        if cfg!(feature = "gpu-vulkan") {
            if let Ok(device) = Self::new(ComputeBackend::Vulkan) {
                return Ok(device);
            }
        }
        
        // Try WGPU
        if cfg!(feature = "wgpu") {
            if let Ok(device) = Self::new(ComputeBackend::Wgpu) {
                return Ok(device);
            }
        }
        
        // No GPU available - strict no-fallback
        Err(ModelError::Backend {
            message: "No GPU backend available. \
                     Compile with --features gpu-cuda, gpu-metal, gpu-vulkan, or wgpu".to_string(),
        })
    }
}
```

---

### 2. RichardsGLU Fused Kernel
**Current State**:
- File: `src/domain/compute/richards_glu_fused_kernel.rs` (exists but incomplete)
- Issue: Operations still separate, not truly fused
- Missing: Device-specific implementations (CUDA kernel, Metal shader, WGPU shader)

**Performance Target**:
- Input: [batch=1K, input_dim=768]
- CPU implementation: ~50ms
- GPU target: ~2ms (25x speedup)
- Memory: Zero-copy (stay on GPU between passes)

**Algorithm**:
```
Pass 1 (Activation):
  hidden = activation(input @ W1 + b1)
  
Pass 2 (Projection):
  output = (hidden * (hidden * W2_gating + b2_gating)) @ W2_projection + b2
  
Fused operation:
  1. Upload input to GPU
  2. Compute hidden = input @ W1 + b1
  3. Apply activation (Richards curve)
  4. Compute gated = hidden * sigmoid(hidden @ W_gate)
  5. Project: output = gated @ W2 + b2
  6. Download output
```

---

### 3. Attention GPU Kernel
**Current State**:
- File: `src/domain/layers/components/temporal_processing_gpu.rs`
- Issue: Skeleton only, no implementation
- Missing: QKV projection, scaled softmax, value projection, output projection

**Performance Target**:
- Input: [batch=512, seq_len=128, embed_dim=768]
- CPU implementation: ~30ms
- GPU target: ~1ms (30x speedup)

**Algorithm**:
```
1. Q = input @ W_q        [batch*seq, embed] @ [embed, embed] → [batch*seq, embed]
2. K = input @ W_k        [batch*seq, embed] @ [embed, embed] → [batch*seq, embed]
3. V = input @ W_v        [batch*seq, embed] @ [embed, embed] → [batch*seq, embed]
4. Reshape to [batch, seq, heads, head_dim]
5. scores = Q @ K^T / √d_head  [batch, heads, seq, seq]
6. scores = softmax(scores)    [batch, heads, seq, seq]
7. context = scores @ V        [batch, heads, seq, head_dim]
8. Reshape back
9. output = context @ W_o      [batch*seq, embed] @ [embed, embed] → [batch*seq, embed]
```

---

### 4. Mamba Selective Scan GPU Kernel
**Current State**:
- File: `src/domain/layers/ssm/components/selective_scan.rs`
- Issue: CPU-only, no GPU variant
- Missing: Selective scan kernel, state propagation

**Performance Target**:
- Input: [batch=512, seq_len=2048, state_dim=256]
- CPU implementation: ~40ms
- GPU target: ~2ms (20x speedup)

**Key Challenge**: Selective scan is inherently sequential per token, but:
- Batch dimension can be parallelized
- Multiple tokens can be processed in parallel within state updates
- Need efficient state memory layout for reuse across tokens

---

## Immediate Actions (Next 2 Hours)

### Action 1: Consolidate GpuDevice Auto-Detection
**File**: `src/domain/compute/gpu_device.rs`

```rust
// Add this implementation
impl GpuDevice {
    /// Create a GPU device with automatic backend detection.
    /// 
    /// Priority order: CUDA > Metal > Vulkan > WGPU
    /// 
    /// # Errors
    /// Returns error if no GPU backend is available on the system.
    /// Strict no-fallback: CPU is not tried.
    pub fn auto_detect() -> Result<Self> {
        // Try backends in priority order
        #[cfg(feature = "gpu-cuda")]
        if let Ok(device) = Self::new(ComputeBackend::Cuda) {
            return Ok(device);
        }
        
        #[cfg(feature = "gpu-metal")]
        if let Ok(device) = Self::new(ComputeBackend::Metal) {
            return Ok(device);
        }
        
        #[cfg(feature = "gpu-vulkan")]
        if let Ok(device) = Self::new(ComputeBackend::Vulkan) {
            return Ok(device);
        }
        
        #[cfg(any(feature = "wgpu", feature = "gpu-wgpu"))]
        if let Ok(device) = Self::new(ComputeBackend::Wgpu) {
            return Ok(device);
        }
        
        // No GPU available
        Err(ModelError::Backend {
            message: "No GPU backend detected. \
                     Available backends: gpu-cuda, gpu-metal, gpu-vulkan, wgpu. \
                     Compile with at least one GPU feature enabled.".to_string(),
        })
    }
}
```

**Testing**:
```rust
#[test]
fn test_auto_detect_respects_priority() {
    // This would require setting up mock backends
    // For now, just verify it works on actual hardware
    match GpuDevice::auto_detect() {
        Ok(device) => {
            println!("Auto-detected: {}", device.backend().as_str());
            assert!(device.backend().is_gpu());
        }
        Err(e) => println!("No GPU (expected on CPU-only): {}", e),
    }
}
```

---

### Action 2: Standardize Lock Handling Pattern
**Search**: All uses of `.device.lock()` in `src/domain/layers/components/`

```bash
# Find all lock patterns
grep -r "\.device\.lock()" src/domain/layers/components/
```

**Refactor Pattern**:
```rust
// Before
let mut device = self.device.lock().map_err(|_| ModelError::Backend {
    message: "Failed to acquire GPU device lock".to_string(),
})?;

// After
let mut device = self.device.lock().map_err(|_| ModelError::Backend {
    message: "GPU device lock failed for [ComponentName]::[MethodName]".to_string(),
})?;
```

---

### Action 3: Audit All GPU Operations for Workspace Pooling
**Files to Audit**:
- `src/domain/layers/components/attention_context_gpu.rs`
- `src/domain/layers/components/feedforward_gpu.rs`
- `src/domain/layers/components/temporal_processing_gpu.rs`

**Checklist**:
- [ ] All `device.allocate()` calls use workspace pool
- [ ] No ad-hoc allocations in kernel implementations
- [ ] `workspace.ensure_capacity()` called before operations
- [ ] `workspace.reset()` called between operations (not deallocate)
- [ ] Memory pool reuse rate tracked and > 99%

---

## Verification Commands

### Check Feature Flags
```bash
cargo build --release --features gpu-all
```

### Test GPU Detection
```bash
cargo test test_auto_detect_no_fallback --lib --features gpu-all
```

### Benchmark GPU vs CPU
```bash
cargo bench --bench gpu_kernels_bench --features gpu-all
```

### Validate Numerical Accuracy
```bash
cargo test gpu_shared_components --lib --features gpu-all
```

---

## Success Criteria (This Session)

By end of session, we should have:

✓ GpuDevice auto-detection consolidated and tested  
✓ All lock handling patterns standardized  
✓ All GPU operations using workspace pooling (> 99% reuse)  
✓ Strict no-fallback semantics enforced everywhere  
✓ Numerical validation tests passing (< 1e-4 tolerance)  
✓ Performance targets documented and verified  

---

## Risk Assessment

### High Risk
- Fused kernel implementations: Require device-specific code (CUDA, Metal, WGPU)
- Performance targets: May not be achievable on all backends
- Numerical accuracy: GPU floating point can differ slightly from CPU

### Medium Risk
- Memory pool integration: Requires careful buffer lifecycle management
- Backend consolidation: Risk of breaking existing GPU functionality
- Testing: Coverage may be incomplete on systems without GPUs

### Low Risk
- Error handling standardization
- Lock pattern refactoring
- Documentation updates

---

## Next Session Handoff

**If continuing this session**:
1. Implement RichardsGLU fused kernel (high impact, 25x speedup)
2. Complete SharedAttentionContext GPU kernel
3. Begin SharedTemporalProcessing attention kernel

**If pausing for break**:
- Auto-detection consolidation is done
- Lock patterns are standardized
- Next session starts with kernel implementation

