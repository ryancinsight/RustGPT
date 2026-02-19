# GPU Consolidation Implementation Roadmap
**Status**: Phase 5.6 - Ready for Implementation  
**Focus**: No-fallback GPU semantics + Fused Kernel Optimization  
**Session Start**: Feb 16, 2026

---

## Phase Structure (3 Parallel Streams)

### Stream 1: GPU Backend Consolidation (2 hours)
**Goal**: Single unified auto-detection with strict error handling

#### Step 1.1: Consolidate GpuDevice::auto_detect()
**File**: `src/domain/compute/gpu_device.rs`

**Current Problem**: Auto-detection logic duplicated across backends

**Solution**:
```rust
impl GpuDevice {
    /// Create GPU device with automatic backend detection.
    /// 
    /// Tries backends in priority order: CUDA > Metal > Vulkan > WGPU
    /// 
    /// # Errors
    /// Returns error if no GPU available (strict no-fallback).
    pub fn auto_detect() -> Result<Self> {
        let backends_to_try = vec![
            (ComputeBackend::Cuda, "gpu-cuda"),
            (ComputeBackend::Metal, "gpu-metal"),
            (ComputeBackend::Vulkan, "gpu-vulkan"),
            (ComputeBackend::Wgpu, "wgpu"),
        ];
        
        for (backend, feature) in backends_to_try {
            // Check if feature is enabled at compile time
            if !Self::is_feature_enabled(feature) {
                continue;
            }
            
            match Self::new(backend) {
                Ok(device) => {
                    log::info!("Auto-detected GPU backend: {}", backend.as_str());
                    return Ok(device);
                }
                Err(_) => continue,  // Try next backend
            }
        }
        
        // No GPU available - strict error (no CPU fallback)
        Err(ModelError::Backend {
            message: format!(
                "No GPU backend detected. Compile with one of:\n  \
                 --features gpu-cuda (NVIDIA)\n  \
                 --features gpu-metal (Apple)\n  \
                 --features gpu-vulkan (AMD/Intel/Linux)\n  \
                 --features wgpu (cross-platform)"
            ),
        })
    }
    
    #[inline]
    fn is_feature_enabled(feature: &str) -> bool {
        cfg!(all(feature = "feature")) == (feature == "feature")  // Pseudo-code
    }
}
```

**Testing**:
```rust
#[test]
fn test_auto_detect_priority_order() {
    // Test that CUDA is tried before Metal, etc.
    // (Requires mock backends or actual GPU hardware)
    match GpuDevice::auto_detect() {
        Ok(device) => {
            let backend = device.backend().as_str();
            println!("Selected backend: {}", backend);
            // Verify priority order in actual implementation
        }
        Err(e) => println!("No GPU (expected on CPU-only): {}", e),
    }
}

#[test]
fn test_auto_detect_no_fallback() {
    // If no GPU available, should error (not fallback to CPU)
    // This test would only pass on CPU-only systems
    match GpuDevice::auto_detect() {
        Ok(_) => assert!(cfg!(any(feature = "gpu-cuda", feature = "gpu-metal"))),
        Err(e) => assert!(e.to_string().contains("GPU backend detected")),
    }
}
```

---

#### Step 1.2: Standardize Lock Handling
**Files to Update**: All files using `.device.lock()`

**Pattern**:
```rust
// ❌ OLD: Generic message
let mut device = self.device.lock().map_err(|_| ModelError::Backend {
    message: "Failed to acquire GPU device lock".to_string(),
})?;

// ✓ NEW: Context-specific message
let mut device = self.device.lock().map_err(|_| ModelError::Backend {
    message: "GPU device lock failed in SharedAttentionContext::forward_gpu".to_string(),
})?;
```

**Refactoring Steps**:
1. Find all `.device.lock()` calls:
   ```bash
   grep -r "\.device\.lock()" src/domain/layers/components/ | grep -v "gpu\\.rs"
   ```

2. For each file, add context to error message
3. Run tests to verify behavior unchanged

---

#### Step 1.3: Consolidate Memory Pool Implementations
**Files to Merge**: 
- `src/domain/compute/cuda/memory.rs`
- `src/domain/compute/metal/memory.rs`
- `src/domain/compute/wgpu_ops.rs` (memory section)

**Current State**: Duplicated power-of-2 sizing logic

**Unified Approach**:
```rust
// In gpu_memory.rs (new unified module)

pub struct GpuMemoryPool {
    capacity: usize,
    element_size: usize,
    next_power_of_two: usize,
}

impl GpuMemoryPool {
    /// Calculate next power-of-2 capacity
    pub fn next_capacity(min_elements: usize, element_size: usize) -> usize {
        let min_bytes = min_elements * element_size;
        let next_pow2 = min_bytes.next_power_of_two();
        next_pow2 / element_size  // Round to nearest power-of-2 elements
    }
}
```

**Verification**: All backends use identical sizing logic

---

### Stream 2: Shared Component GPU Implementation (4 hours)

#### Step 2.1: SharedAttentionContext GPU Kernel
**File**: `src/domain/layers/components/attention_context_gpu.rs`

**Current State**: File exists with stubs

**Implementation Plan**:

**2.1.1: Define Workspace Structure**
```rust
#[derive(Debug)]
pub struct AttentionContextGpuWorkspace {
    /// Input tensor: [batch_size, embed_dim]
    buf_input: GpuBuffer,
    /// Context matrix: [embed_dim, embed_dim]
    buf_context: GpuBuffer,
    /// Intermediate similarity scores: [batch_size, embed_dim]
    buf_scores: GpuBuffer,
    /// Output tensor: [batch_size, embed_dim]
    buf_output: GpuBuffer,
    /// Capacity tracking
    capacity: (usize, usize),  // (batch_size, embed_dim)
    /// Allocation statistics
    allocation_count: usize,
}

impl AttentionContextGpuWorkspace {
    pub fn new() -> Self {
        Self {
            buf_input: GpuBuffer::null(),
            buf_context: GpuBuffer::null(),
            buf_scores: GpuBuffer::null(),
            buf_output: GpuBuffer::null(),
            capacity: (0, 0),
            allocation_count: 0,
        }
    }
    
    /// Ensure workspace has capacity (power-of-2 sizing)
    pub fn ensure_capacity(
        &mut self,
        device: &mut GpuDevice,
        batch_size: usize,
        embed_dim: usize,
    ) -> Result<()> {
        if batch_size <= self.capacity.0 && embed_dim <= self.capacity.1 {
            return Ok(());
        }
        
        // Calculate power-of-2 capacity
        let new_batch = batch_size.next_power_of_two().max(2);
        let new_embed = embed_dim.next_power_of_two().max(2);
        
        // Allocate buffers
        let bytes = std::mem::size_of::<f32>();
        self.buf_input = device.allocate(new_batch * new_embed * bytes)?;
        self.buf_context = device.allocate(new_embed * new_embed * bytes)?;
        self.buf_scores = device.allocate(new_batch * new_embed * bytes)?;
        self.buf_output = device.allocate(new_batch * new_embed * bytes)?;
        
        self.capacity = (new_batch, new_embed);
        self.allocation_count += 1;
        
        Ok(())
    }
}
```

**2.1.2: GPU Forward Implementation**
```rust
impl SharedAttentionContext {
    pub fn forward_gpu(&mut self, input: &Array2<f32>, strength: f32) -> Result<Array2<f32>> {
        let mut device = self.gpu_device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in SharedAttentionContext::forward_gpu".to_string(),
        })?;
        
        let (batch_size, embed_dim) = input.dim();
        
        // Ensure workspace has capacity
        let workspace = self.workspace.as_mut().ok_or(ModelError::Backend {
            message: "GPU workspace not initialized".to_string(),
        })?;
        workspace.ensure_capacity(&mut device, batch_size, embed_dim)?;
        
        // Step 1: Upload input and context to GPU
        device.upload(input.as_slice().unwrap(), &mut workspace.buf_input)?;
        if let Some(ctx) = self.incoming_context.as_ref() {
            device.upload(ctx.as_slice().unwrap(), &mut workspace.buf_context)?;
        }
        
        // Step 2: Compute similarity scores
        // scores = input @ context^T (scaled)
        device.gemm_f32(
            strength,  // Scale by context strength
            &workspace.buf_input,
            &workspace.buf_context,  // Note: Matrix dimensions transposed logically
            0.0,
            &mut workspace.buf_scores,
            batch_size,
            embed_dim,
            embed_dim,
            false,  // Don't transpose A
            true,   // Transpose B (context^T)
        )?;
        
        // Step 3: Apply softmax normalization
        device.softmax(&workspace.buf_scores, &mut workspace.buf_output)?;
        
        // Step 4: Combine with input (residual + weighted context)
        // output = input + output
        device.add(&workspace.buf_input, &workspace.buf_output, &mut workspace.buf_output)?;
        
        // Step 5: Download result
        let mut output = vec![0.0f32; batch_size * embed_dim];
        device.download(&workspace.buf_output, &mut output)?;
        
        // Update statistics
        self.stats.kernel_launches += 4;  // GEMM, softmax, add, download
        self.stats.bytes_uploaded += (batch_size * embed_dim + embed_dim * embed_dim) * 4;
        self.stats.bytes_downloaded += batch_size * embed_dim * 4;
        
        Ok(Array2::from_shape_vec((batch_size, embed_dim), output)?)
    }
}
```

**2.1.3: Integration with CPU Path**
```rust
impl SharedAttentionContext {
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // GPU path (strict no-fallback if GPU is set)
        if self.gpu_device.is_some() {
            return self.forward_gpu(input, self.similarity_context_strength[[0, 0]])
                .expect("GPU forward failed in SharedAttentionContext - no CPU fallback");
        }
        
        // CPU path (when GPU not available)
        self.forward_cpu(input)
    }
    
    fn forward_cpu(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Existing CPU implementation...
        // (unchanged)
    }
}
```

**Testing**:
```rust
#[test]
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn test_attention_context_gpu_vs_cpu() {
    let input = Array2::random((128, 512));
    let mut context_component = SharedAttentionContext::new();
    
    // CPU reference
    let cpu_output = context_component.forward_cpu(&input);
    
    // GPU computation (if available)
    if let Ok(mut gpu_component) = SharedAttentionContext::auto_detect() {
        gpu_component.set_input_context(input.clone());
        let gpu_output = gpu_component.forward_gpu(&input, 1.0).expect("GPU forward");
        
        // Verify numerical equivalence (< 1e-4 relative error)
        for (cpu, gpu) in cpu_output.iter().zip(gpu_output.iter()) {
            let rel_error = (cpu - gpu).abs() / (cpu.abs().max(1e-6));
            assert!(rel_error < 1e-4, "Error: {}", rel_error);
        }
    }
}
```

---

#### Step 2.2: SharedFeedforward RichardsGLU Fused Kernel
**File**: `src/domain/layers/components/feedforward_gpu.rs`

**Target Speedup**: 25x (50ms → 2ms on 1K batch)

**Algorithm**: Two-pass fused kernel avoiding intermediate downloads

**2.2.1: Workspace Structure**
```rust
#[derive(Debug)]
pub struct FeedforwardGpuWorkspace {
    buf_input: GpuBuffer,
    buf_w1: GpuBuffer,
    buf_b1: GpuBuffer,
    buf_hidden: GpuBuffer,
    buf_w2: GpuBuffer,
    buf_b2: GpuBuffer,
    buf_output: GpuBuffer,
    capacity: (usize, usize, usize),  // (batch, input_dim, hidden_dim)
}
```

**2.2.2: Fused Kernel Pseudo-code**
```
KERNEL richards_glu_fused(input, w1, b1, w2, b2):
  // Pass 1: Projection + Activation
  FOR each row in input:
    hidden = input[row] @ w1 + b1
    hidden = richards_curve(hidden)
  
  // Pass 2: Gating + Final Projection
  FOR each row in hidden:
    gate = sigmoid(hidden[row] @ w_gate)
    gated = hidden[row] * gate
    output[row] = gated @ w2 + b2
```

**2.2.3: Rust Implementation**
```rust
impl SharedFeedforward {
    pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let mut device = self.gpu_device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in SharedFeedforward::forward_gpu".to_string(),
        })?;
        
        let (batch_size, input_dim) = input.dim();
        let (_, output_dim) = self.w2.dim();
        let hidden_dim = self.w1.dim().1;
        
        // Ensure workspace capacity
        let workspace = self.workspace.as_mut().ok_or(ModelError::Backend {
            message: "Workspace not initialized".to_string(),
        })?;
        workspace.ensure_capacity(&mut device, batch_size, input_dim, hidden_dim)?;
        
        // Upload inputs (only once at start)
        device.upload(input.as_slice().unwrap(), &mut workspace.buf_input)?;
        device.upload(self.w1.as_slice().unwrap(), &mut workspace.buf_w1)?;
        device.upload(self.b1.as_slice().unwrap(), &mut workspace.buf_b1)?;
        device.upload(self.w2.as_slice().unwrap(), &mut workspace.buf_w2)?;
        device.upload(self.b2.as_slice().unwrap(), &mut workspace.buf_b2)?;
        
        // PASS 1: hidden = richards(input @ W1 + b1)
        device.gemm_f32(
            1.0, &workspace.buf_input, &workspace.buf_w1,
            0.0, &mut workspace.buf_hidden,
            batch_size, hidden_dim, input_dim,
            false, false
        )?;
        
        // Add bias (on GPU, not CPU)
        device.add_bias(&workspace.buf_hidden, &workspace.buf_b1)?;
        
        // Apply Richards activation (on GPU, not CPU)
        device.richards(&workspace.buf_hidden, &mut workspace.buf_hidden)?;
        
        // PASS 2: output = (hidden * gate) @ W2 + b2
        // For simplicity, assume gate = 1 for now
        device.gemm_f32(
            1.0, &workspace.buf_hidden, &workspace.buf_w2,
            0.0, &mut workspace.buf_output,
            batch_size, output_dim, hidden_dim,
            false, false
        )?;
        
        // Add final bias (on GPU)
        device.add_bias(&workspace.buf_output, &workspace.buf_b2)?;
        
        // Download result (single download at end)
        let mut output = vec![0.0f32; batch_size * output_dim];
        device.download(&workspace.buf_output, &mut output)?;
        
        // Stats
        self.stats.kernel_launches += 5;  // 2 GEMM + bias + activation + download
        self.stats.bytes_uploaded += (input_dim + hidden_dim + output_dim) * 4;
        self.stats.bytes_downloaded += batch_size * output_dim * 4;
        
        Ok(Array2::from_shape_vec((batch_size, output_dim), output)?)
    }
}
```

**Verification**: Benchmark against CPU implementation
```bash
cargo bench --bench feedforward_bench --features gpu-all
# Target: 25x speedup on 1K batch
```

---

#### Step 2.3: SharedTemporalProcessing GPU Kernels
**File**: `src/domain/layers/components/temporal_processing_gpu.rs`

**Three kernels to implement**:

**A. Attention Kernel (30x speedup)**
```rust
impl SharedTemporalProcessing {
    pub fn attention_forward_gpu(
        &mut self,
        input: &Array2<f32>,
        params: &AttentionParams,
    ) -> Result<Array2<f32>> {
        // QKV projection → scaled softmax → output projection
        // See unified_gpu_kernels.rs for reference
    }
}
```

**B. Mamba Selective Scan (20x speedup)**
```rust
pub fn mamba_scan_forward_gpu(
    &mut self,
    input: &Array2<f32>,
    state: &mut Array1<f32>,
) -> Result<Array2<f32>> {
    // Token-by-token selective scan with GPU-accelerated matrix ops
    // State persists across tokens
}
```

**C. RG-LRU Recurrent (15x speedup)**
```rust
pub fn rglru_forward_gpu(
    &mut self,
    input: &Array2<f32>,
    state: &mut Array1<f32>,
) -> Result<Array2<f32>> {
    // Recurrent computation: output[t] = activation(input[t] @ W + state[t-1] @ U)
}
```

---

### Stream 3: Testing & Validation (2 hours)

#### Step 3.1: Numerical Validation Test Suite
**File**: `tests/gpu_shared_components_phase56.rs`

**Template**:
```rust
#[test]
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn test_component_gpu_vs_cpu_numerical() {
    let batch_sizes = vec![1, 32, 128, 512, 1024];
    
    for batch_size in batch_sizes {
        let input = Array2::random((batch_size, 512));
        
        let mut cpu_component = ComponentUnderTest::new();
        let cpu_output = cpu_component.forward(&input);
        
        let mut gpu_component = ComponentUnderTest::auto_detect().expect("GPU required");
        let gpu_output = gpu_component.forward(&input).expect("GPU forward");
        
        // Verify numerical equivalence
        verify_relative_error(&cpu_output, &gpu_output, 1e-4);
    }
}

fn verify_relative_error(expected: &Array2<f32>, actual: &Array2<f32>, tolerance: f32) {
    for (e, a) in expected.iter().zip(actual.iter()) {
        let rel_err = (e - a).abs() / (e.abs().max(1e-6));
        assert!(rel_err < tolerance, 
            "Relative error {} exceeds tolerance {}", rel_err, tolerance);
    }
}
```

---

#### Step 3.2: Performance Benchmark Suite
**File**: `benches/gpu_kernels_bench.rs`

**Template**:
```rust
#[bench]
fn bench_attention_context_gpu(b: &mut Bencher) {
    let input = Array2::random((512, 768));
    let mut component = SharedAttentionContext::auto_detect().unwrap();
    
    b.iter(|| {
        component.forward_gpu(&input, 1.0).unwrap()
    });
    
    // Expected: ~1-5ms (vs 30ms CPU = 6-30x speedup)
}

#[bench]
fn bench_feedforward_gpu(b: &mut Bencher) {
    let input = Array2::random((1024, 768));
    let mut component = SharedFeedforward::auto_detect().unwrap();
    
    b.iter(|| {
        component.forward_gpu(&input).unwrap()
    });
    
    // Expected: ~2ms (vs 50ms CPU = 25x speedup)
}
```

---

## Success Metrics Per Stream

### Stream 1 Completion (2 hours)
- ✓ `GpuDevice::auto_detect()` consolidated and tested
- ✓ All lock patterns use context-specific error messages
- ✓ Memory pool implementations merged across backends
- ✓ `cargo test --lib --features gpu-all` passes

### Stream 2 Completion (4 hours)
- ✓ AttentionContext GPU kernel fully implemented (70% → 100%)
- ✓ SharedFeedforward RichardsGLU fused kernel (40% → 100%)
- ✓ SharedTemporalProcessing Attention/Mamba kernels (20% → 80%)
- ✓ All kernels use workspace pooling (zero ad-hoc allocates)
- ✓ All activations computed on GPU (not CPU post-download)

### Stream 3 Completion (2 hours)
- ✓ Numerical tests pass: GPU vs CPU < 1e-4 error across batch sizes
- ✓ Benchmarks show >= 15x speedup for all kernels
- ✓ Integration test: full pipeline (Attention → Feedforward → Temporal)
- ✓ Auto-detection works and is tested

---

## Build & Test Commands

### Compile with GPU support
```bash
# All GPU backends
cargo build --release --features gpu-all

# Specific backend
cargo build --release --features gpu-cuda
cargo build --release --features gpu-metal
cargo build --release --features wgpu
```

### Run validation tests
```bash
cargo test --lib gpu_shared_components --features gpu-all -- --nocapture

cargo test test_attention_context_gpu_vs_cpu --lib --features gpu-all

cargo test test_feedforward_gpu_vs_cpu --lib --features gpu-all
```

### Run benchmarks
```bash
cargo bench --bench gpu_kernels_bench --features gpu-all

# With output
cargo bench --bench gpu_kernels_bench --features gpu-all -- --verbose
```

### Check diagnostics
```bash
# Verify auto-detection
cargo test test_auto_detect_no_fallback --lib

# Check memory usage
cargo test test_workspace_reuse_efficiency --lib --features gpu-all

# Verify numerical accuracy
cargo test gpu_numerical_validation --lib --features gpu-all -- --nocapture
```

---

## Deployment Checklist

Before considering Phase 5.6 complete:

- [ ] `GpuDevice::auto_detect()` works on CUDA/Metal/WGPU
- [ ] No code duplication between backend implementations
- [ ] All GPU operations use strict no-fallback error handling
- [ ] Memory pool reuse rate > 99%
- [ ] SharedAttentionContext GPU kernel passes numerical tests
- [ ] SharedFeedforward RichardsGLU fused kernel achieves 25x speedup
- [ ] SharedTemporalProcessing supports all 3 temporal types (GPU)
- [ ] Integration test passes (full component pipeline)
- [ ] Benchmarks document performance targets met
- [ ] Documentation updated with GPU usage examples
- [ ] CI/CD green with all GPU features enabled

---

## Next Session Preparation

If pausing work:

1. **Save current state**: All GPU detection and consolidation code complete
2. **Document**: Incomplete kernel implementations with clear TODO markers
3. **Test status**: Unit tests passing, integration tests pending
4. **Benchmark**: Baseline measurements ready, targets defined
5. **Handoff**: Quick reference guide + diagnostic report prepared

Resume by:
1. Running `cargo test --lib --features gpu-all`
2. Reading `QUICK_REFERENCE_GPU_CONSOLIDATION_FEB16.md`
3. Focusing on RichardsGLU kernel (highest impact)
4. Running benchmarks after each kernel completion

