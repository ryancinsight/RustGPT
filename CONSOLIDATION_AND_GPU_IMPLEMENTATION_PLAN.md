# Consolidation & GPU Backend Implementation Plan
**Phase 5.3 - Shared Component Optimization & GPU Integration**

**Date**: February 13, 2026  
**Status**: Foundation Complete → Active Implementation  
**Approach**: Strict no-fallback GPU detection with automatic backend selection

---

## Overview

This session consolidates memory management and workspace patterns across Diffusion, SSM (Mamba/RG-LRU), and Transformer architectures, while beginning GPU backend kernel implementation. The strategy enforces strict GPU requirements—no silent CPU fallback—to prevent performance cliffs during development.

---

## Architecture State Analysis

### Current Memory & Workspace Patterns

| Component | Location | Current State | Consolidation Target |
|-----------|----------|--------|----------------------|
| **Transformer** | `src/domain/layers/transformer/` | `TransformerBlockStreamingWorkspace` (private buffers) | `UnifiedLayerWorkspace` (shared) |
| **Mamba** | `src/domain/layers/ssm/mamba.rs` | `MambaStreamingWorkspace` (state-specific) | `UnifiedLayerWorkspace` (+ streaming state) |
| **RG-LRU** | `src/domain/layers/ssm/rg_lru.rs` | `RgLruStreamingWorkspace` (variant-specific) | `UnifiedLayerWorkspace` (+ streaming state) |
| **Diffusion** | `src/domain/layers/diffusion/` | Ad-hoc buffers + `SharedAttentionContext` | `UnifiedLayerWorkspace` (+ diffusion fields) |
| **Shared Attention** | `src/domain/layers/components/attention_context.rs` | Supports Transformer + Diffusion | Integrate into `UnifiedLayerWorkspace` GPU support |
| **Shared Feedforward** | `src/domain/layers/components/feedforward.rs` | Shared across all architectures | GPU kernel variants via `GpuMatrixOps` |
| **GPU Memory** | `src/domain/compute/gpu_memory.rs` | Abstract trait + WGPU impl | CUDA/Metal stub → full kernels |

### Redundancy Points Identified

1. **Per-architecture workspace duplication** (Transformer, Mamba, RG-LRU all define similar buffer structures)
2. **Diffusion lacks unified workspace** (mixes `SharedAttentionContext` with local allocations)
3. **GPU ops trait defined but not integrated** into components (SharedFeedforward, RichardsNorm, etc.)
4. **Power-of-2 buffer sizing** implemented independently in multiple places

---

## Phase 5.3 Consolidation Objectives

### Stage 1: Unified Workspace Integration (Week 1)

**Goal**: Replace all architecture-specific `*StreamingWorkspace` with `UnifiedLayerWorkspace` across Transformer, Diffusion, Mamba, and RG-LRU.

#### 1.1 UnifiedLayerWorkspace Enhancements

**Current structure** (lines 37-85 in `unified_layer_workspace.rs`):
- Core buffers: `norm1_out`, `temporal_out`, `residual1`, `feedforward_in`, `feedforward_out`, etc.
- GPU buffers: `gpu_norm1_out`, `gpu_temporal_out`, etc. (feature-gated)

**Changes needed**:

1. **Conditional streaming state fields** (for SSM architectures):
   ```rust
   pub struct UnifiedLayerWorkspace {
       // Existing shared buffers
       norm1_out: Option<Array2<f32>>,
       // ...
       
       // Streaming state (populated only when SSM architecture is active)
       #[cfg(feature = "streaming")]
       pub streaming_state: Option<StreamingState>,
       
       // Diffusion-specific buffers
       #[cfg(feature = "diffusion")]
       pub diffusion_buffers: Option<DiffusionWorkspaceBuffers>,
   }
   ```

2. **GPU buffer variants** for all intermediate results:
   ```rust
   #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal", feature = "gpu-wgpu"))]
   pub struct GpuWorkspaceBuffers {
       pub norm1_out: Option<GpuBuffer>,
       pub temporal_out: Option<GpuBuffer>,
       pub residual1: Option<GpuBuffer>,
       // ... all CPU buffers have GPU equivalents
   }
   ```

#### 1.2 Diffusion Block GPU Integration

**Update**: `src/domain/layers/diffusion/block.rs`

Replace loose buffer allocations with `UnifiedLayerWorkspace` + `DiffusionWorkspaceBuffers`.

```rust
pub struct DiffusionBlock {
    pub unified_workspace: UnifiedLayerWorkspace,
    pub diffusion_buffers: DiffusionWorkspaceBuffers,
    // ... existing fields
}

impl DiffusionBlock {
    pub fn forward_gpu(
        &mut self,
        input: &Array2<f32>,
        time_embed: &Array2<f32>,
        gpu_device: &mut GpuDevice,
    ) -> Result<Array2<f32>> {
        // 1. Upload input to GPU
        let gpu_input = gpu_device.ops.upload(input)?;
        
        // 2. Allocate outputs in workspace
        self.unified_workspace.ensure_capacity(input.nrows(), input.ncols())?;
        
        // 3. Execute GPU ops via workspace
        // ...
        
        // 4. Download results to CPU
        let result = gpu_device.ops.download(&gpu_result)?;
        Ok(result)
    }
}
```

#### 1.3 Transformer Block GPU Integration

**Update**: `src/domain/layers/transformer/block.rs`

Replace `TransformerBlockStreamingWorkspace` with calls to `UnifiedLayerWorkspace`.

```rust
pub struct TransformerBlock {
    pub unified_workspace: UnifiedLayerWorkspace,
    // Remove TransformerBlockStreamingWorkspace
}
```

#### 1.4 SSM Block GPU Integration

**Updates**: `src/domain/layers/ssm/mamba.rs` and `rg_lru.rs`

Add GPU-aware state management:

```rust
pub struct MambaBlock {
    pub unified_workspace: UnifiedLayerWorkspace,
    pub streaming_state: Option<StreamingState>,
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal", feature = "gpu-wgpu"))]
    pub gpu_state: Option<GpuStreamingState>,
}
```

---

### Stage 2: GPU Kernel Implementation (Week 2-3)

**Goal**: Implement core GPU kernels for WGPU, CUDA, and Metal backends.

#### 2.1 WGPU Kernels (Primary Implementation)

**Location**: `src/domain/compute/wgpu_ops.rs`

**Priority 1: GEMM (Tiled Matrix Multiplication)**

```wgsl
// shader/gemm_f32.wgsl
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct MatrixDims {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
}

@group(1) @binding(0) var<uniform> dims: MatrixDims;

const TILE_SIZE: u32 = 16u;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    
    var sum: f32 = 0.0;
    for (var k_tile: u32 = 0u; k_tile < dims.k; k_tile += TILE_SIZE) {
        var tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
        var tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;
        
        let local_id = @builtin(local_invocation_id);
        let local_x = local_id.x;
        let local_y = local_id.y;
        
        // Load tile from A
        let a_idx = row * dims.lda + (k_tile + local_y);
        if (a_idx < arrayLength(&A)) {
            tile_a[local_x][local_y] = A[a_idx];
        }
        
        // Load tile from B
        let b_idx = (k_tile + local_x) * dims.ldb + col;
        if (b_idx < arrayLength(&B)) {
            tile_b[local_x][local_y] = B[b_idx];
        }
        
        workgroupBarrier();
        
        // Compute partial sum
        for (var k: u32 = 0u; k < TILE_SIZE; k++) {
            sum += tile_a[local_x][k] * tile_b[k][local_y];
        }
        
        workgroupBarrier();
    }
    
    let c_idx = row * dims.ldc + col;
    if (c_idx < arrayLength(&C)) {
        C[c_idx] = sum;
    }
}
```

Implementation checklist:
- [ ] Load WGPU shader modules
- [ ] Bind buffer groups (A, B, C matrices + dims uniform)
- [ ] Dispatch compute shader with proper workgroup sizing
- [ ] Add synchronization for tiled computation
- [ ] Test correctness against CPU baseline

**Priority 2: Element-Wise Operations**

**Activations** (ReLU, GELU, SiLU):
```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        output[idx] = max(input[idx], 0.0);
    }
}

fn gelu(x: f32) -> f32 {
    return x * 0.5 * (1.0 + tanh(sqrt(2.0 / 3.14159) * (x + 0.044715 * x * x * x)));
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}
```

**Priority 3: Layer Normalization**

Two-pass kernel:
1. **Pass 1**: Compute mean and variance
2. **Pass 2**: Normalize and scale

```wgsl
@compute @workgroup_size(256)
fn layer_norm_pass1(...) {
    // Parallel reduction to compute mean & variance
}

@compute @workgroup_size(256)
fn layer_norm_pass2(...) {
    // Normalize: (x - mean) / sqrt(var + eps)
    // Scale: normalized * weight + bias
}
```

**Priority 4: Softmax (Numerically Stable)**

Log-sum-exp trick:
```wgsl
@compute @workgroup_size(256)
fn softmax_pass1(...) {
    // Find max per row
}

@compute @workgroup_size(256)
fn softmax_pass2(...) {
    // Compute log-sum-exp
}

@compute @workgroup_size(256)
fn softmax_pass3(...) {
    // Apply: exp(x - max - log_sum_exp)
}
```

#### 2.2 CUDA Kernels

**Location**: `src/domain/compute/cuda/ops.rs`

Implement using cuBLAS + custom kernels via `cudarc`:

```rust
impl GpuMatrixOps for CudaMatrixOps {
    fn gemm_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &mut GpuBuffer,
    ) -> Result<()> {
        // Use cuBLAS cublasGemm for efficiency
        self.device.gemm(m, n, k, a, b, c)?;
        Ok(())
    }
    
    fn relu(&self, input: &GpuBuffer, output: &mut GpuBuffer) -> Result<()> {
        // Custom kernel or CUDA library call
        unsafe {
            self.device.launch_on_stream(
                self.relu_kernel.clone(),
                GridBlock { grid: (64, 1, 1), block: (256, 1, 1) },
                (input.id, output.id),
            )?;
        }
        Ok(())
    }
}
```

#### 2.3 Metal Kernels (macOS Support)

**Location**: `src/domain/compute/metal/ops.rs`

Use Metal Performance Shaders (MPS) or custom Metal shaders:

```rust
impl GpuMatrixOps for MetalMatrixOps {
    fn gemm_f32(&self, ...) -> Result<()> {
        let cmd_buffer = self.command_queue.new_command_buffer();
        let mm_descriptor = metal::MTLMatrixMultiplyDescriptor::new();
        // Use MPSMatrixMultiplication
        Ok(())
    }
}
```

---

### Stage 3: Component GPU Integration (Week 3-4)

**Goal**: Add GPU variants to core shared components.

#### 3.1 SharedAttentionContext GPU Support

**Update**: `src/domain/layers/components/attention_context.rs`

```rust
impl SharedAttentionContext {
    /// GPU-accelerated attention with strict no-fallback
    pub fn apply_context_gpu(
        &mut self,
        activation: &Array2<f32>,
        gpu_device: &mut GpuDevice,
    ) -> Result<Array2<f32>> {
        // 1. Upload activation to GPU
        let gpu_activation = gpu_device.ops.upload(activation)?;
        
        // 2. Compute similarity via GEMM
        let gpu_similarity = gpu_device.ops.gemm_f32(
            activation.nrows(),
            activation.nrows(),
            activation.ncols(),
            &gpu_activation,
            &gpu_activation,
            // output buffer from workspace
        )?;
        
        // 3. Apply softmax
        gpu_device.ops.softmax(&gpu_similarity)?;
        
        // 4. Download result
        gpu_device.ops.download(&gpu_similarity)
    }
}
```

#### 3.2 SharedFeedforward GPU Support

**Update**: `src/domain/layers/components/feedforward.rs`

```rust
impl SharedFeedforward {
    pub fn forward_gpu(
        &mut self,
        input: &Array2<f32>,
        workspace: &mut UnifiedLayerWorkspace,
        gpu_device: &mut GpuDevice,
    ) -> Result<Array2<f32>> {
        workspace.ensure_gpu_capacity(
            input.nrows(),
            self.hidden_dim,
        )?;
        
        // 1. Project to hidden: output = input @ W_hidden
        let gpu_hidden = gpu_device.ops.gemm_f32(
            input.nrows(),
            self.hidden_dim,
            input.ncols(),
            &gpu_input,
            &gpu_weight_hidden,
        )?;
        
        // 2. Add bias
        gpu_device.ops.axpy(
            1.0,
            &gpu_bias_hidden,
            &mut gpu_hidden,
        )?;
        
        // 3. Apply activation (SiLU or ReLU depending on GLU variant)
        gpu_device.ops.silu(&gpu_hidden)?;
        
        // 4. Project back to output
        let gpu_output = gpu_device.ops.gemm_f32(
            input.nrows(),
            input.ncols(),
            self.hidden_dim,
            &gpu_hidden,
            &gpu_weight_output,
        )?;
        
        gpu_device.ops.download(&gpu_output)
    }
}
```

#### 3.3 RichardsNorm GPU Support

**Update**: `src/domain/layers/components/richards_norm.rs`

```rust
impl RichardsNorm {
    pub fn forward_gpu(
        &mut self,
        input: &Array2<f32>,
        gpu_device: &mut GpuDevice,
    ) -> Result<Array2<f32>> {
        // 1. Layer norm on GPU
        let gpu_normalized = gpu_device.ops.layer_norm(
            &gpu_input,
            &gpu_weight,
            &gpu_bias,
            self.eps,
        )?;
        
        // 2. Apply Richards activation (fused with norm)
        gpu_device.ops.custom("richards_activation", &gpu_normalized)?;
        
        gpu_device.ops.download(&gpu_normalized)
    }
}
```

---

### Stage 4: Integration & Testing (Week 4-5)

#### 4.1 TransformerBlock GPU Forward Pass

**File**: `src/domain/layers/transformer/block.rs`

```rust
impl TransformerBlock {
    pub fn forward_gpu(
        &mut self,
        input: &Array2<f32>,
        gpu_device: &mut GpuDevice,
    ) -> Result<Array2<f32>> {
        // Ensure GPU workspace capacity
        self.unified_workspace.ensure_gpu_capacity(
            input.nrows(),
            input.ncols(),
        )?;
        
        // 1. Attention block
        let mut attended = self.apply_attention_gpu(input, gpu_device)?;
        
        // 2. Residual connection
        gpu_device.ops.add_scaled(1.0, input, 1.0, &mut attended)?;
        
        // 3. FFN block
        let output = self.apply_feedforward_gpu(&attended, gpu_device)?;
        
        // 4. Residual connection
        gpu_device.ops.add_scaled(1.0, &attended, 1.0, &mut output)?;
        
        Ok(output)
    }
}
```

#### 4.2 DiffusionBlock GPU Forward Pass

**File**: `src/domain/layers/diffusion/block.rs`

```rust
impl DiffusionBlock {
    pub fn forward_gpu(
        &mut self,
        noisy_x: &Array2<f32>,
        time_embed: &Array2<f32>,
        gpu_device: &mut GpuDevice,
    ) -> Result<Array2<f32>> {
        // Similar pattern to TransformerBlock
        // + FiLM modulation via GPU
    }
}
```

#### 4.3 GPU-Aware Model Training Loop

**Update**: `src/application/training/llm_training.rs`

```rust
pub async fn train_with_gpu(
    model: &mut LLMModel,
    batch: &TrainingBatch,
    config: &TrainingConfig,
) -> Result<TrainingMetrics> {
    // Initialize GPU device with auto-detection
    let mut gpu_device = GpuDevice::auto_detect()
        .expect("GPU required in strict mode");
    
    // Forward pass on GPU
    let logits = model.forward_gpu(&batch.input, &mut gpu_device)?;
    
    // Compute loss on GPU
    let loss = compute_cross_entropy_gpu(&logits, &batch.target, &mut gpu_device)?;
    
    // Backward pass on GPU (gradients computed in place)
    model.backward_gpu(&loss, &mut gpu_device)?;
    
    // Update weights on GPU
    model.update_weights_gpu(&config.learning_rate, &mut gpu_device)?;
    
    Ok(TrainingMetrics { loss })
}
```

---

### Stage 5: Automatic GPU Detection & No-Fallback Enforcement (Week 5)

**File**: `src/domain/compute/gpu_device.rs`

Enhance auto-detection logic:

```rust
impl GpuDevice {
    /// Detect available GPU with strict requirements
    pub fn auto_detect() -> Result<Self> {
        if let Ok(backend_env) = std::env::var("RUSTGPT_GPU_BACKEND") {
            match backend_env.as_str() {
                "auto-gpu" => Self::detect_with_priority(),
                "cuda" if cfg!(feature = "gpu-cuda") => Self::cuda_required()?,
                "metal" if cfg!(feature = "gpu-metal") => Self::metal_required()?,
                "wgpu" if cfg!(feature = "gpu-wgpu") => Self::wgpu_required()?,
                _ => Err("Invalid RUSTGPT_GPU_BACKEND value".into()),
            }
        } else {
            Self::detect_with_priority()
        }
    }
    
    /// Priority detection: CUDA > Metal > WGPU (no CPU fallback)
    fn detect_with_priority() -> Result<Self> {
        #[cfg(feature = "gpu-cuda")]
        if let Ok(device) = Self::cuda_required() {
            return Ok(device);
        }
        
        #[cfg(feature = "gpu-metal")]
        if let Ok(device) = Self::metal_required() {
            return Ok(device);
        }
        
        #[cfg(feature = "gpu-wgpu")]
        if let Ok(device) = Self::wgpu_required() {
            return Ok(device);
        }
        
        Err("No GPU backend available. Compile with --features gpu-wgpu (recommended)".into())
    }
}
```

---

## Implementation Commands

### Build & Test Commands

**CPU-only (default)**:
```bash
cargo build --release
cargo test --lib
```

**WGPU (cross-platform GPU)**:
```bash
cargo build --release --features gpu-wgpu
cargo test --lib --features gpu-wgpu
```

**CUDA (requires NVIDIA GPU + CUDA toolkit)**:
```bash
cargo build --release --features gpu-cuda
cargo test --lib --features gpu-cuda
```

**All backends**:
```bash
cargo build --release --features gpu-all
cargo test --lib --features gpu-all
```

**GPU with strict no-fallback**:
```bash
RUSTGPT_GPU_BACKEND=auto-gpu cargo run --release --features gpu-wgpu
```

**Require specific backend**:
```bash
RUSTGPT_GPU_BACKEND=cuda cargo run --release --features gpu-cuda
RUSTGPT_GPU_BACKEND=metal cargo run --release --features gpu-metal
```

---

## Performance Targets

| Operation | CPU Baseline | GPU Target | Speedup |
|-----------|--------------|------------|---------|
| GEMM (4096×4096) | 150ms | 5ms | 30× |
| ReLU (1M elements) | 8ms | 0.5ms | 16× |
| LayerNorm (1M elements) | 12ms | 1ms | 12× |
| Softmax (1M elements) | 15ms | 1.5ms | 10× |
| Full forward pass (seq_len=1024, model_dim=2048) | 500ms | 25ms | 20× |

---

## Testing Strategy

### Unit Tests
- GPU kernel correctness vs. CPU baselines (tolerance: ±1e-5)
- Memory allocation/deallocation tracking
- Feature flag combinations

### Integration Tests
- Full model forward pass on GPU
- Training step with GPU backward pass
- Diffusion inference on GPU
- SSM streaming with GPU state

### Stress Tests
- Large batch sizes (256, 512)
- Extended sequences (up to 8192 tokens)
- Memory fragmentation under repeated allocation

---

## Success Criteria

**Phase 5.3 Completion**:
1. ✅ All architectures use `UnifiedLayerWorkspace` (no per-arch duplicates)
2. ✅ WGPU GEMM kernel implemented and tested
3. ✅ Core activations (ReLU, GELU, SiLU) on GPU
4. ✅ LayerNorm on GPU with 10× speedup min
5. ✅ Softmax on GPU
6. ✅ SharedFeedforward GPU variants
7. ✅ SharedAttentionContext GPU support
8. ✅ TransformerBlock GPU integration
9. ✅ DiffusionBlock GPU integration
10. ✅ Auto-detection with no CPU fallback when GPU backend selected
11. ✅ 517 tests passing (maintained)
12. ✅ GPU tests skip gracefully if no device available

---

## References

- **Previous Status**: `GPU_BACKEND_IMPLEMENTATION_STATUS.md`
- **Consolidation Context**: Thread @T-019c572e-210d-727f-8b51-6342e0b79988
- **Attention Context**: `src/domain/layers/components/attention_context.rs`
- **Unified Workspace**: `src/domain/layers/components/unified_layer_workspace.rs`
- **GPU Abstractions**: `src/domain/compute/gpu_device.rs`, `gpu_memory.rs`, `gpu_ops.rs`
