# Immediate Action Plan - GPU Consolidation Phase 5.6b

## Priority Stack (by effort vs. impact)

### 1. **HIGHEST PRIORITY: SharedFeedforward GpuComponent Implementation** 
**Status**: 🔴 Not Started
**Effort**: 3-4 hours
**Impact**: Unlocks GPU feedforward in Transformer & Diffusion

#### Step 1.1: Check Current Implementation
```rust
// Location: src/domain/layers/components/feedforward.rs
// Need to find:
// - Current struct definition
// - Existing CPU forward path
// - GPU cache/buffers if any
```

#### Step 1.2: Implement GpuComponent Trait
```rust
pub struct SharedFeedforward { ... }

impl GpuComponent for SharedFeedforward {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) { ... }
    fn enable_gpu_auto_detect(&mut self) -> Result<()> { ... }
    fn is_gpu_ready(&self) -> bool { ... }
    fn gpu_backend_name(&self) -> Option<&'static str> { ... }
    fn gpu_device(&self) -> Option<Arc<Mutex<GpuDevice>>> { ... }
    fn ensure_capacity(&mut self, batch_size: usize, embed_dim: usize, seq_len: usize) -> Result<()> { ... }
}
```

#### Step 1.3: Replace `feedforward_gpu.rs` Placeholder
**Current Implementation** (lines 32-97):
- `forward_gpu_richards()` → falls back to CPU (line 48)
- `forward_gpu_moe()` → falls back to CPU (line 73)
- `forward_gpu_dispatch()` → orchestrates dispatch

**Required Changes**:
```rust
pub fn forward_gpu_richards(
    &mut self,
    input: &Array2<f32>,
    ctx: &mut GpuSharedOpsContext,
    ops: &mut dyn GpuMatrixOps,
) -> Result<Array2<f32>> {
    // 1. Upload input to GPU
    let input_gpu = ctx.upload_buffer(input)?;
    
    // 2. Call x1, x2 = linear_split(input)  // 2 GEMMs
    let mut x1 = ctx.allocate_buffer(...)?;
    let mut x2 = ctx.allocate_buffer(...)?;
    ops.gemm_f32(ctx.pool(), 1.0, &input_gpu, &w1, 0.0, &mut x1, ...)?;
    ops.gemm_f32(ctx.pool(), 1.0, &input_gpu, &w2, 0.0, &mut x2, ...)?;
    
    // 3. Apply Richards curve activation
    let mut x2_activated = ctx.allocate_buffer(...)?;
    ops.richards_curve(ctx.pool(), &x2, &mut x2_activated, params)?;
    
    // 4. Element-wise multiply x1 * x2_activated
    let mut output = ctx.allocate_buffer(...)?;
    ops.mul(ctx.pool(), &x1, &x2_activated, &mut output, ...)?;
    
    // 5. Download result to CPU
    ctx.download_buffer(&output)
}
```

#### Step 1.4: Testing
- Create `test_feedforward_gpu_forward()` - verify output matches CPU within ε ≤ 1e-4
- Create `test_feedforward_gpu_memory_efficiency()` - verify no allocations during forward
- Create `test_feedforward_gpu_moe_dispatch()` - test MoE variant

---

### 2. **HIGH PRIORITY: SharedTemporalProcessing GPU Kernels**
**Status**: 🔴 Not Started
**Effort**: 8-10 hours
**Impact**: Unlocks attention variants on GPU

#### Step 2.1: PolyAttention GPU Kernel (3 hours)
**File**: `src/domain/layers/components/temporal_processing_gpu.rs`, lines 64-81

```rust
pub fn forward_gpu_poly_attention(...) -> Result<Array2<f32>> {
    // Current: Returns input.clone() (line 80)
    // Required:
    // 1. Q = input @ W_q (GEMM)
    // 2. K = input @ W_k (GEMM)
    // 3. V = input @ W_v (GEMM)
    // 4. scores = Q @ K.T  (GEMM)
    // 5. Apply polynomial basis expansion if configured
    // 6. gates = softmax(scores) (GPU kernel)
    // 7. output = gates @ V (GEMM)
    // 8. Apply output projection
}
```

#### Step 2.2: Mamba/RG-LRU GPU Kernel (4-5 hours)
**File**: `src/domain/layers/components/temporal_processing_gpu.rs`, lines 92-133

For Mamba:
```rust
pub fn forward_gpu_mamba(...) -> Result<Array2<f32>> {
    // 1. x_proj = input @ A (GEMM)
    // 2. Conv1D(x_proj) → causal convolution (GPU kernel)
    // 3. h = SSM(s, h_prev) → selective scan (GPU kernel)
    // 4. LayerNorm(h)
    // 5. output = h * (input @ B) (GEMM)
}
```

For RG-LRU:
```rust
pub fn forward_gpu_rg_lru(...) -> Result<Array2<f32>> {
    // 1. x_proj = input @ W_in (GEMM)
    // 2. forget = sigmoid(x_proj @ W_f + b_f) (GPU activation)
    // 3. value = tanh(x_proj @ W_v + b_v) (GPU activation)
    // 4. h = forget * h_prev + (1 - forget) * value (GPU element-wise)
    // 5. output = LayerNorm(h) * (h @ W_out) (GEMM)
}
```

#### Step 2.3: Testing
- Each variant: output matches CPU within ε ≤ 1e-4
- Causal masking tests for attention variants
- State management tests for SSM variants

---

### 3. **MEDIUM PRIORITY: SharedAttentionContext GPU Optimization**
**Status**: 🟡 Partially Complete
**Effort**: 2-3 hours
**Impact**: Faster attention context updates

**File**: `src/domain/layers/components/attention_context_gpu.rs`, lines 46-125

**Current Status**:
- Methods exist: `apply_incoming_context_gpu()`, `update_outgoing_context_gpu()`
- Implementation status unclear, likely placeholders

**Required**:
- Kernel fusion for context matrix operations
- Test for memory efficiency

---

## Implementation Template

### GpuComponent Trait Implementation Template
```rust
// In the component struct
pub struct SharedFeedforward {
    // ... existing fields ...
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,
    gpu_stats: GpuExecutionStats,
}

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
        self.gpu_device.as_ref().map(|d| {
            let locked = d.lock().unwrap();
            locked.backend_name()
        })
    }

    fn gpu_device(&self) -> Option<Arc<Mutex<GpuDevice>>> {
        self.gpu_device.clone()
    }

    fn ensure_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        seq_len: usize,
    ) -> Result<()> {
        // Pre-allocate buffers for the given dimensions
        // This is called before forward pass to avoid allocations during compute
        if let Some(device) = &self.gpu_device {
            let locked = device.lock().unwrap();
            // Allocate buffers based on dimensions
            // e.g., w1_buffer, w2_buffer, w_out_buffer, etc.
        }
        Ok(())
    }
}
```

---

## Validation Checklist

For each component GPU implementation:

- [ ] **Compilation**
  - [ ] Builds without errors
  - [ ] No unused variables warnings
  - [ ] Tests compile with `--features gpu-wgpu`

- [ ] **Correctness**
  - [ ] Output matches CPU implementation within ε ≤ 1e-4
  - [ ] Tested with multiple batch sizes
  - [ ] Tested with edge cases (small dims, large dims)

- [ ] **Memory Efficiency**
  - [ ] No allocations during forward pass (all pre-allocated)
  - [ ] Power-of-2 sizing verified
  - [ ] Pool stats show high reuse rate

- [ ] **Performance**
  - [ ] Runs on GPU (not CPU fallback)
  - [ ] GEMM operations hit 50-100+ TFLOPS
  - [ ] Latency < CPU implementation

- [ ] **Error Handling**
  - [ ] Explicit errors when GPU not available
  - [ ] No panic on GPU operations
  - [ ] Clear error messages for troubleshooting

---

## Next Session Kickoff

**If starting fresh next session**:
1. Read this file + `CONSOLIDATION_GPU_PHASE5.6_SESSION_STATUS.md`
2. Start with SharedFeedforward (simplest, most isolated)
3. Use the template above as copy-paste starting point
4. Test after each step (don't implement all methods at once)
5. Verify with: `cargo test --lib` (should pass all 539 tests + new GPU tests)

**Session Goals** (realistically achievable):
- SharedFeedforward: GpuComponent impl + replace placeholder (3-4 hours)
- SharedTemporalProcessing: PolyAttention GPU kernel (3-4 hours)
- Testing & validation (2-3 hours)
- **Total: 8-11 hours → unlock functional GPU acceleration**

---

## Key References

### Related Files to Review
- Example WGPU kernel: `src/domain/compute/wgpu_ops.rs` (lines for GEMM, activation kernels)
- Trait definition: `src/domain/compute/gpu_component.rs`
- Example usage: `src/domain/richards/richards_glu.rs` (GPU path, lines 180-260)

### GPU Operation Available
From `GpuMatrixOps` trait:
- `gemm_f32()` - General matrix multiply
- `gemv_f32()` - Matrix-vector multiply
- `relu()`, `gelu()`, `silu()`, `sigmoid()`, `tanh()` - Activations
- `softmax()` - Softmax
- `layer_norm()` - Layer normalization
- `richards_curve()` - Richards activation
- `mul()`, `add_scaled()`, `scale()`, `axpy()` - Element-wise
- And 15+ more specialized operations

### Memory Pool
From `GpuMemoryPool` trait:
- `allocate(size)` - Pre-allocate buffer
- `upload(data)` - CPU → GPU
- `download(buffer)` - GPU → CPU
- `copy_within_device()` - GPU → GPU
- Power-of-2 sizing automatically applied
