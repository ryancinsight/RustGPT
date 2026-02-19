# Phase 5.3 Immediate Actions
**Start Date**: February 13, 2026  
**Session Focus**: Consolidation Foundation + GPU Kernel Kickoff

---

## Priority 1: UnifiedLayerWorkspace GPU Enhancement (TODAY)

### 1.1 Extend UnifiedLayerWorkspace for GPU

**File**: `src/domain/layers/components/unified_layer_workspace.rs`

**Action**: Add feature-gated GPU buffer variants and consolidate capacity tracking

```rust
// Pseudo-code - integrate into existing struct

#[cfg(any(feature = "gpu-cuda", feature = "gpu-metal", feature = "gpu-wgpu"))]
pub struct GpuWorkspaceBuffers {
    pub norm1_out: Option<GpuBuffer>,
    pub temporal_out: Option<GpuBuffer>,
    pub residual1: Option<GpuBuffer>,
    pub feedforward_in: Option<GpuBuffer>,
    pub feedforward_out: Option<GpuBuffer>,
    pub attention_out: Option<GpuBuffer>,
}

pub struct UnifiedLayerWorkspace {
    // Existing CPU buffers
    norm1_out: Option<Array2<f32>>,
    temporal_out: Option<Array2<f32>>,
    // ... rest
    
    // New GPU buffers (Phase 5.3)
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal", feature = "gpu-wgpu"))]
    gpu_buffers: Option<GpuWorkspaceBuffers>,
    
    // Unified capacity tracking
    last_batch_size: usize,
    last_feature_dim: usize,
}

impl UnifiedLayerWorkspace {
    /// Ensure GPU buffers exist and have required capacity
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal", feature = "gpu-wgpu"))]
    pub fn ensure_gpu_capacity(
        &mut self,
        batch_size: usize,
        feature_dim: usize,
        gpu_memory: &mut dyn GpuMemoryPool,
    ) -> Result<()> {
        let new_capacity = batch_size * feature_dim * std::mem::size_of::<f32>();
        
        if self.gpu_buffers.is_none() {
            self.gpu_buffers = Some(GpuWorkspaceBuffers::new(gpu_memory, new_capacity)?);
        }
        
        // Track for subsequent allocations
        self.last_batch_size = batch_size;
        self.last_feature_dim = feature_dim;
        
        Ok(())
    }
}
```

**Verification**:
```bash
cargo test --lib unified_layer_workspace --features gpu-wgpu
```

---

### 1.2 Add GPU Buffer Consolidation Test

**File**: `tests/gpu_workspace_integration.rs` (NEW)

```rust
#[cfg(feature = "gpu-wgpu")]
mod tests {
    use llm::domain::layers::components::UnifiedLayerWorkspace;
    use llm::domain::compute::GpuDevice;
    
    #[test]
    fn test_unified_workspace_gpu_capacity() {
        let mut workspace = UnifiedLayerWorkspace::new();
        let mut gpu_device = GpuDevice::auto_detect().expect("GPU required");
        
        // Allocate for batch_size=32, feature_dim=2048
        workspace.ensure_gpu_capacity(32, 2048, &mut gpu_device.memory).unwrap();
        
        // Verify allocation
        let stats = gpu_device.memory.memory_stats();
        assert!(stats.allocated > 0, "GPU memory should be allocated");
        assert!(stats.allocated >= 32 * 2048 * 4, "Allocation too small");
    }
}
```

**Status**: ⏳ Pending

---

## Priority 2: SharedAttentionContext GPU Support (TODAY)

### 2.1 Add GPU Forward Method

**File**: `src/domain/layers/components/attention_context.rs`

**Action**: Implement `apply_context_gpu()` method with strict no-fallback

```rust
impl SharedAttentionContext {
    /// GPU-accelerated attention context application
    /// 
    /// Requires GPU backend to be available; panics if GPU not selected.
    #[cfg(any(feature = "gpu-cuda", feature = "gpu-metal", feature = "gpu-wgpu"))]
    pub fn apply_context_gpu(
        &mut self,
        activation: &ndarray::ArrayView2<f32>,
        gpu_device: &mut crate::domain::compute::GpuDevice,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        // Step 1: Ensure workspace allocated
        let batch_size = activation.nrows();
        let hidden_dim = activation.ncols();
        
        if self.outgoing_context.is_none() 
            || self.outgoing_context.as_ref().unwrap().shape() != [hidden_dim, hidden_dim] {
            self.outgoing_context = Some(Array2::zeros((hidden_dim, hidden_dim)));
        }
        
        // Step 2: Upload activations to GPU
        let gpu_activation = gpu_device.ops.upload(activation)?;
        
        // Step 3: Compute similarity matrix via GEMM
        // similarity = activation @ activation.T
        let similarity_shape = (batch_size, batch_size);
        let mut gpu_similarity = gpu_device.memory.allocate(
            similarity_shape.0 * similarity_shape.1 * std::mem::size_of::<f32>()
        )?;
        
        gpu_device.ops.gemm_f32(
            batch_size,      // M
            batch_size,      // N
            hidden_dim,      // K
            &gpu_activation,
            &gpu_activation,
            &mut gpu_similarity,
        )?;
        
        // Step 4: Apply softmax (numerically stable)
        gpu_device.ops.softmax(&mut gpu_similarity)?;
        
        // Step 5: Download result
        let result = gpu_device.ops.download(&gpu_similarity)?;
        
        // Step 6: Cleanup
        gpu_device.memory.deallocate(gpu_similarity);
        
        Ok(result)
    }
}
```

**Verification**:
```bash
cargo test --lib attention_context --features gpu-wgpu
```

**Status**: ⏳ Pending

---

## Priority 3: Diffusion Block GPU Workspace Integration (TODAY)

### 3.1 Examine Current Diffusion Buffer Usage

**File**: `src/domain/layers/diffusion/block.rs`

**Action**: Identify buffer allocation patterns and map to UnifiedLayerWorkspace

```bash
grep -n "Array2::zeros\|Option<Array2" src/domain/layers/diffusion/block.rs
```

Expected output shows:
- Local buffer allocations in `forward()`
- Repeated allocations per block

**Consolidation goal**: Replace with workspace-managed buffers

---

### 3.2 Refactor DiffusionBlock to Use UnifiedLayerWorkspace

**File**: `src/domain/layers/diffusion/block.rs`

**Changes**:
1. Add `unified_workspace: UnifiedLayerWorkspace` field
2. Replace inline `Array2::zeros()` calls with workspace buffer reuse
3. Implement `ensure_capacity()` pattern

```rust
pub struct DiffusionBlock {
    pub unified_workspace: UnifiedLayerWorkspace,  // NEW
    
    // Existing fields
    pub pre_attention_norm: RichardsNorm,
    pub temporal_mixing: SharedTemporalProcessing,
    pub pre_ffn_norm: RichardsNorm,
    pub feedforward: SharedFeedforward,
    pub film_modulation: SharedFilmModulation,
    
    // Remove per-block buffer fields
}

impl DiffusionBlock {
    pub fn forward(
        &mut self,
        noisy_x: &Array2<f32>,
        time_embed: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let batch_size = noisy_x.nrows();
        let embed_dim = noisy_x.ncols();
        
        // Ensure workspace capacity once per block
        self.unified_workspace.ensure_capacity(batch_size, embed_dim)?;
        
        // Use workspace buffers instead of allocating inline
        let norm_out = self.unified_workspace.norm1_out.get_or_insert_with(
            || Array2::zeros((batch_size, embed_dim))
        );
        
        // Apply operations...
        
        Ok(output)
    }
}
```

**Status**: ⏳ Pending

---

## Priority 4: WGPU GEMM Kernel Skeleton (THIS WEEK)

### 4.1 Create WGPU Shader Directory Structure

**Action**: Organize shader files

```bash
mkdir -p src/domain/compute/wgpu/shaders
touch src/domain/compute/wgpu/shaders/mod.rs
touch src/domain/compute/wgpu/shaders/gemm.wgsl
touch src/domain/compute/wgpu/shaders/activation.wgsl
touch src/domain/compute/wgpu/shaders/layer_norm.wgsl
touch src/domain/compute/wgpu/shaders/softmax.wgsl
```

### 4.2 Implement Basic GEMM Shader

**File**: `src/domain/compute/wgpu/shaders/gemm.wgsl`

```wgsl
// Tiled matrix multiplication shader for GEMM
// Computes: C = A @ B (F32 format)

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
    
    // Bounds check
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Iterate over k dimension in tiles
    for (var k_tile: u32 = 0u; k_tile < dims.k; k_tile += TILE_SIZE) {
        // Load A[row, k_tile:k_tile+TILE_SIZE]
        for (var k: u32 = 0u; k < TILE_SIZE; k++) {
            if (k_tile + k < dims.k) {
                let a_idx = row * dims.lda + (k_tile + k);
                let a_val = A[a_idx];
                
                // Load B[k_tile+k, col]
                let b_idx = (k_tile + k) * dims.ldb + col;
                let b_val = B[b_idx];
                
                sum += a_val * b_val;
            }
        }
    }
    
    let c_idx = row * dims.ldc + col;
    C[c_idx] = sum;
}
```

### 4.3 Integrate Shader into WgpuOps

**File**: `src/domain/compute/wgpu_ops.rs`

**Action**: Add shader loading and dispatch

```rust
impl WgpuOps {
    fn load_gemm_shader(&mut self) -> Result<()> {
        let shader_source = include_str!("wgpu/shaders/gemm.wgsl");
        self.gemm_shader = Some(self.device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some("gemm_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            }
        ));
        Ok(())
    }
    
    pub fn gemm_f32(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &mut GpuBuffer,
    ) -> Result<()> {
        // 1. Load shader if not cached
        if self.gemm_shader.is_none() {
            self.load_gemm_shader()?;
        }
        
        // 2. Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("gemm_pipeline"),
                layout: None,
                module: self.gemm_shader.as_ref().unwrap(),
                entry_point: "main",
                compilation_options: Default::default(),
            }
        );
        
        // 3. Create bind groups
        // TODO: Implement bind group creation
        
        // 4. Dispatch compute shader
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            
            let workgroup_x = (m as u32 + 15) / 16;
            let workgroup_y = (n as u32 + 15) / 16;
            cpass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        Ok(())
    }
}
```

**Status**: ⏳ Pending

---

## Priority 5: Compile & Test with GPU Features

### 5.1 Verify Feature Compilation

**Action**: Build with each feature combination

```bash
# CPU only (should pass)
cargo build --release

# WGPU (cross-platform GPU)
cargo build --release --features gpu-wgpu

# All backends
cargo build --release --features gpu-all

# Run tests
cargo test --lib --features gpu-wgpu
```

**Expected result**: All 517 tests pass

**Status**: ⏳ Pending

---

### 5.2 Verify No-Fallback Enforcement

**Action**: Test that GPU backend selection panics on missing kernels

```bash
# This should work:
RUSTGPT_GPU_BACKEND=auto-gpu cargo run --release --features gpu-wgpu --example check_gpu

# This should panic with clear error if kernel incomplete:
# (Implementation needed)
```

**Status**: ⏳ Pending

---

## Session Roadmap

### Session 1 (Today - Feb 13)
- [ ] Extend UnifiedLayerWorkspace for GPU support
- [ ] Add GPU forward to SharedAttentionContext
- [ ] Refactor DiffusionBlock to use unified workspace
- [ ] Create WGPU shader directory structure
- [ ] Implement basic GEMM shader
- [ ] Verify compilation with all feature combinations
- [ ] Update GPU_BACKEND_IMPLEMENTATION_STATUS.md

### Session 2 (Feb 14)
- [ ] Implement WGPU activation kernels (ReLU, GELU, SiLU)
- [ ] Implement WGPU layer normalization
- [ ] Add layer norm GPU tests
- [ ] Integrate SharedFeedforward GPU support
- [ ] Begin CUDA stub → kernel migration

### Session 3 (Feb 15)
- [ ] Implement WGPU softmax (numerically stable)
- [ ] Add data transfer kernels (upload/download/copy)
- [ ] Integrate SharedAttentionContext into TransformerBlock GPU
- [ ] Test full transformer block GPU forward pass

### Session 4 (Feb 16)
- [ ] Integrate DiffusionBlock GPU forward pass
- [ ] Test diffusion GPU inference
- [ ] Implement CUDA core kernels (GEMM, activations)
- [ ] Performance profiling and bottleneck analysis

### Session 5 (Feb 17)
- [ ] Metal backend kernels
- [ ] SSM GPU streaming state support
- [ ] End-to-end training with GPU
- [ ] Performance optimization and tuning

---

## Blocking Issues & Mitigations

| Issue | Mitigation |
|-------|-----------|
| WGPU shader compilation errors | Validate WGSL syntax separately before integration |
| GPU device not available | Skip GPU tests gracefully; ensure feature gates work |
| Buffer synchronization | Explicit barriers after compute; test with smaller matrices first |
| Performance regression | Compare against CPU baseline for each kernel |

---

## Success Metrics (Session 1)

✅ = Complete  
⏳ = In Progress  
❌ = Blocked

- [ ] UnifiedLayerWorkspace supports GPU buffers
- [ ] SharedAttentionContext has GPU forward method
- [ ] DiffusionBlock refactored to use unified workspace
- [ ] WGPU GEMM shader compiles and loads
- [ ] All 517 tests passing with --features gpu-wgpu
- [ ] No CPU fallback when GPU backend selected
- [ ] Documentation updated

---

## References

- Plan: `CONSOLIDATION_AND_GPU_IMPLEMENTATION_PLAN.md` (this session)
- Status: `GPU_BACKEND_IMPLEMENTATION_STATUS.md`
- Previous thread: @T-019c572e-210d-727f-8b51-6342e0b79988
