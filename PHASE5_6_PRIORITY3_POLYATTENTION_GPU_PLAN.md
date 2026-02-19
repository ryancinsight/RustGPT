# Priority 3: PolyAttention GPU Implementation Plan

**Status**: Starting Implementation  
**Priority**: 3 (High Impact)  
**Estimated Effort**: 4-6 hours  
**Expected Speedup**: 30x for 512-batch

---

## 1. PolyAttention Current State

### Structure
```rust
pub struct PolyAttention {
    // Learnable parameters
    w_q: Array2<f32>,  // Query projection (embed_dim × embed_dim)
    w_k: Array2<f32>,  // Key projection (embed_dim × embed_dim)
    w_v: Array2<f32>,  // Value projection (embed_dim × embed_dim)
    w_out: Array2<f32>, // Output projection (embed_dim × embed_dim)
    
    // Polynomial activation parameters
    a: Array2<f32>,    // Scalar coefficient (1×1)
    b: Array2<f32>,    // Scalar bias (1×1)
    scale: Array2<f32>, // Scalar scale (1×1)
    p: usize,          // Degree (e.g., 3, 5)
    
    // Mixture-of-Heads gating
    moh: MoHGating,    // Head selection & routing
    
    // Contextual Positional Embeddings
    cope: UnifiedCoPE,
    window_size: Option<usize>,
    
    // GPU weights (for future use)
    gpu_weights: Option<PolyAttentionGpuWeights>,
}
```

### CPU Forward Path
**File**: `poly_attention.rs:2775+`
1. Project input: Q = input @ w_q, K = input @ w_k, V = input @ w_v
2. Reshape to heads: (batch, seq_len, num_heads, head_dim)
3. For each head:
   - Compute content scores: Q @ K^T
   - Add CoPE distance embeddings
   - Apply polynomial activation: scale * (a * x^p + b)
   - Apply MoH gating: select top-k heads
   - Compute attention: softmax(scores) @ V
4. Concatenate head outputs
5. Project output: concat @ w_out

**Computational Complexity**: O(batch * seq_len^2 * embed_dim) for full attention

---

## 2. GPU Kernel Strategy

### Two-Level Fusion

**Level 1: Per-Head Kernels** (simplest, Phase 1)
```
For each head:
  1. Content scores: Q_h @ K_h^T → (batch, seq_len, seq_len)
  2. Add CoPE: scores += cope_embeddings
  3. Polynomial activation: y = scale * (a * smooth_clip(x, 8)^p + b)
  4. Softmax: norm(y)
  5. Aggregate: softmax @ V_h
  6. Accumulate to output
```
**Launches**: 6 per head (total: 6 * num_heads)

**Level 2: Fused Multi-Head Kernel** (optimized, Phase 2)
```
Single kernel processes all heads in parallel:
  1. Batched content scores: Q @ K^T for all heads at once
  2. Vectorized polynomial activation across all heads
  3. Parallel softmax & aggregation
```
**Launches**: 3 total (content scores, poly activation, attention)

---

## 3. GPU Kernel Components

### Kernel 1: Content Scores & CoPE
**Input**: Q_h (batch, seq_len, head_dim), K_h (batch, seq_len, head_dim), positions
**Output**: scores (batch, seq_len, seq_len)

```wgsl
// Pseudo-code
for (b, i, j) in (batch, seq_len, seq_len):
    // Content score
    score = dot(Q_h[b, i], K_h[b, j])
    
    // Add CoPE distance embedding
    dist = i - j
    cope_val = cope_embeddings[dist]
    score += cope_val
    
    scores[b, i, j] = score
```

**GPU Type**: Matrix multiply + element-wise addition  
**Cost**: 1 GEMM (Q @ K^T) + 1 element-wise add  
**Optimal Backend**: Prefer GEMM libraries (cuBLAS, Metal BLASonGPU)

### Kernel 2: Polynomial Activation
**Input**: scores (batch, seq_len, seq_len)
**Output**: activated (batch, seq_len, seq_len)

```wgsl
// For each element in scores
fn poly_activation(x: f32) -> f32 {
    // Smooth clip to avoid overflow
    clipped = smooth_clip(x, 8.0)
    
    // Polynomial: a * x^p + b
    poly = a * pow(clipped, p) + b
    
    // Scale output
    return scale * poly
}

// Apply element-wise
for idx in 0..total_elements:
    activated[idx] = poly_activation(scores[idx])
```

**GPU Type**: Element-wise activation  
**Cost**: 1 GPU kernel (parallel reduction for pow)  
**Optimal Backend**: Custom WGSL kernel

### Kernel 3: Softmax & Attention
**Input**: activated (batch, seq_len, seq_len), V_h (batch, seq_len, head_dim)
**Output**: output (batch, seq_len, head_dim)

```wgsl
// Row-wise softmax on activated scores
for (b, i) in (batch, seq_len):
    max_score = max(activated[b, i, :])
    exp_vals = exp(activated[b, i, :] - max_score)
    sum_exp = sum(exp_vals)
    softmax_weights = exp_vals / sum_exp
    
    # Weighted sum of values
    output[b, i] = softmax_weights @ V_h[b, :, :]
```

**GPU Type**: Softmax reduction + matrix multiply  
**Cost**: 1 softmax kernel + 1 weighted sum  
**Optimal Backend**: Use existing softmax from gpu_ops

### Kernel 4: MoH Gating (Optional GPU Optimization)
**Input**: gating logits (batch, seq_len, num_heads)
**Output**: gate_values (batch, seq_len, num_heads), selected_heads (batch, top_k)

**Current**: CPU-based head selection via threshold predictor  
**GPU Version**: Not critical for Phase 3 - can stay on CPU

---

## 4. Implementation Roadmap

### Phase 3.1: GPU Infrastructure Setup (0.5 hours)
1. Add `gpu_device: Option<Arc<Mutex<GpuDevice>>>` to PolyAttention
2. Implement GpuComponent trait
3. Wire device attachment and auto-detect

**Files to Modify**:
- `src/domain/attention/poly_attention.rs` - Add GPU device field, GpuComponent impl
- `src/domain/attention/poly_attention_gpu.rs` - Implement GPU helper functions

### Phase 3.2: Basic GPU Forward (2 hours)
1. Upload Q, K, V to GPU
2. Implement content scores kernel (use existing GEMM)
3. Implement CoPE addition kernel
4. Implement polynomial activation kernel

**Files to Create/Modify**:
- `src/domain/compute/gpu_device.rs` - Add convenience methods if needed
- `src/domain/compute/wgpu_ops.rs` - Implement poly_attention_fused kernel
- `src/domain/attention/poly_attention.rs` - Add forward_gpu method

### Phase 3.3: Softmax & Aggregation (1.5 hours)
1. Softmax computation on GPU
2. Attention @ V aggregation
3. Head concatenation
4. Output projection

### Phase 3.4: MoH Gating Integration (1 hour)
1. Keep MoH on CPU (threshold predictor is complex)
2. Wire head selection to GPU path
3. Mask unused heads

### Phase 3.5: Testing & Validation (1 hour)
1. Unit test for GPU forward
2. Numerical accuracy validation
3. Benchmarking (CPU vs GPU)

---

## 5. GPU Kernel Dispatch Design

### New Method: `PolyAttention::forward_gpu()`

```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // 1. Get GPU device
    let device_arc = self.gpu_device.as_ref()
        .ok_or(ModelError::Backend { ... })?
        .clone();
    
    let mut device = device_arc.lock().unwrap();
    
    // 2. Upload input & weights to GPU
    let batch_size = input.nrows();
    let seq_len = input.ncols() / self.embed_dim;
    
    let input_gpu = device.upload_f32(input.as_slice().unwrap())?;
    let w_q_gpu = ... // Upload Q, K, V weights
    let w_k_gpu = ...
    let w_v_gpu = ...
    
    // 3. Compute projections: Q, K, V = input @ w_q, w_k, w_v
    let mut q_gpu = device.allocate_f32(batch_size * seq_len * embed_dim)?;
    device.gemm_f32(1.0, &input_gpu, &w_q_gpu, 0.0, &mut q_gpu, ...)?;
    
    // 4. Reshape to heads: (batch, seq_len, num_heads, head_dim)
    let mut q_heads_gpu = reshape_for_heads(q_gpu, ...)?;
    
    // 5. Compute attention for each head
    for head in 0..self.num_heads {
        let q_h = get_head_slice(&q_heads_gpu, head);
        let k_h = get_head_slice(&k_heads_gpu, head);
        let v_h = get_head_slice(&v_heads_gpu, head);
        
        // Content scores + CoPE
        let mut scores = compute_content_scores_gpu(device, &q_h, &k_h)?;
        add_cope_gpu(device, &mut scores, &self.cope, seq_len)?;
        
        // Polynomial activation
        let mut poly_scores = apply_polynomial_activation_gpu(
            device,
            &scores,
            self.a[[0, 0]],
            self.b[[0, 0]],
            self.scale[[0, 0]],
            self.p,
        )?;
        
        // Softmax + Attention
        apply_softmax_gpu(device, &mut poly_scores)?;
        let mut head_output = aggregate_attention_gpu(device, &poly_scores, &v_h)?;
        
        // Accumulate
        accumulate_head_output(device, &mut output, &head_output, head)?;
    }
    
    // 6. Output projection
    let mut output_final = device.allocate_f32(batch_size * seq_len * embed_dim)?;
    device.gemm_f32(1.0, &output, &w_out_gpu, 0.0, &mut output_final, ...)?;
    
    // 7. Download result
    let mut result = Array2::zeros((batch_size, seq_len * embed_dim));
    device.download(&output_final, result.as_slice_mut().unwrap())?;
    
    Ok(result)
}
```

---

## 6. GPU Kernel Traits

### New Trait in `gpu_ops.rs`

```rust
pub trait GpuMatrixOps {
    // ... existing methods ...
    
    // New for PolyAttention
    fn poly_attention_scores(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        q: &GpuBuffer,           // (batch*seq_len, head_dim)
        k: &GpuBuffer,           // (batch*seq_len, head_dim)
        output: &mut GpuBuffer,  // (batch*seq_len, seq_len)
        batch_size: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()>;
    
    fn poly_activation_kernel(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,       // (batch, seq_len, seq_len)
        output: &mut GpuBuffer,  // Same shape
        a: f32,
        b: f32,
        scale: f32,
        p: usize,
    ) -> Result<()>;
    
    fn add_cope_embeddings(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        scores: &mut GpuBuffer,  // (batch*seq_len, seq_len)
        cope: &GpuBuffer,        // Positional embeddings
        seq_len: usize,
    ) -> Result<()>;
}
```

---

## 7. Integration with Shared Components

### PolyAttention as GpuComponent

```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuComponent for PolyAttention {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        // Allocate GPU weights
        let mut dev = device.lock().unwrap();
        self.gpu_weights = Some(PolyAttentionGpuWeights {
            w_q: dev.upload_f32(self.w_q.as_slice().unwrap())?,
            w_k: dev.upload_f32(self.w_k.as_slice().unwrap())?,
            w_v: dev.upload_f32(self.w_v.as_slice().unwrap())?,
            w_out: dev.upload_f32(self.w_out.as_slice().unwrap())?,
        });
        self.gpu_device = Some(device);
    }
    
    fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;
        self.set_gpu_device(Arc::new(Mutex::new(device)));
        Ok(())
    }
    
    fn is_gpu_ready(&self) -> bool {
        self.gpu_device.is_some() && self.gpu_weights.is_some()
    }
    
    fn gpu_backend_name(&self) -> Option<&'static str> {
        self.gpu_device.as_ref()?.lock().ok().map(|d| d.backend().as_str())
    }
    
    fn ensure_capacity(&mut self, batch_size: usize, embed_dim: usize, seq_len: usize) -> Result<()> {
        // Verify dimensions match
        if self.embed_dim != embed_dim {
            return Err(ModelError::InvalidInput { ... });
        }
        Ok(())
    }
}
```

---

## 8. Expected Performance

### Batch=512, seq_len=256, embed_dim=768, num_heads=12

| Operation | CPU Time | GPU Time | Speedup |
|---|---|---|---|
| Q, K, V projections | 15ms | 0.5ms | 30x |
| Content scores (12 heads) | 45ms | 1.5ms | 30x |
| CoPE addition | 5ms | 0.2ms | 25x |
| Poly activation | 8ms | 0.3ms | 27x |
| Softmax + Attention | 20ms | 0.7ms | 29x |
| Output projection | 12ms | 0.4ms | 30x |
| MoH gating | 3ms | 3ms | 1x (stays on CPU) |
| **Total** | **~108ms** | **~6.6ms** | **~16x** |

*Note: Lower speedup (16x vs 24x from RichardsGlu) due to sequence length^2 complexity in attention.*

---

## 9. Files to Modify/Create

### New Files
- (None - use existing GPU infrastructure)

### Modified Files
1. `src/domain/attention/poly_attention.rs`
   - Add `gpu_device: Option<Arc<Mutex<GpuDevice>>>`
   - Implement `GpuComponent` trait
   - Add `forward_gpu()` method
   - Helper functions for GPU operations

2. `src/domain/attention/poly_attention_gpu.rs`
   - Implement GPU helper functions (currently empty stub)
   - Parameter conversion utilities
   - Kernel launch helpers

3. `src/domain/compute/gpu_device.rs`
   - Add convenience methods (if needed for PolyAttention)
   - Ensure GEMM, softmax, element-wise ops are exposed

4. `src/domain/compute/wgpu_ops.rs`
   - Add `poly_attention_scores()` kernel
   - Add `poly_activation_kernel()` kernel
   - Add `add_cope_embeddings()` kernel

---

## 10. Compilation Checklist

- [ ] PolyAttention compiles with GPU features
- [ ] GpuComponent trait implemented correctly
- [ ] GPU device attachment works
- [ ] Parameter upload succeeds
- [ ] GPU forward path compiles
- [ ] Tests pass: `cargo test --lib --features wgpu`
- [ ] No warnings or errors

---

## Next: Implementation

Ready to begin Phase 3.1 (GPU Infrastructure Setup)

**Target**: ~24 hours of development across multiple sessions
**First Session Goal**: Complete Phase 3.1 & 3.2 (Infrastructure + Basic Kernels)
