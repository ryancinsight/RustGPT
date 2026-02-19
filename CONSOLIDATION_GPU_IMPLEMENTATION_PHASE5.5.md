# GPU Backend Consolidation & Optimization - Phase 5.5
**Date**: February 14, 2026  
**Status**: In Progress  
**Focus**: Unified GPU Manager Consolidation + Shared Component GPU Implementation

---

## Phase Objectives

### Primary Goal
Consolidate duplicate GPU managers into a single `UnifiedGpuBufferPool` and implement GPU variants for all shared components (Diffusion, SSM, Transformer) with strict no-fallback mode.

### Success Criteria
- [ ] `UnifiedGpuBufferPool` implemented and tested
- [ ] `SharedComponentGpuManager` and `GpuSharedOpsContext` deprecated (backward compat maintained)
- [ ] All shared components implement `GpuComponent` trait
- [ ] DiffusionBlock has GPU forward path
- [ ] SSM (Mamba/RG-LRU) has GPU recurrent kernels
- [ ] TransformerBlock full end-to-end GPU verified
- [ ] Zero silent fallbacks - all GPU operations explicit error if GPU unavailable
- [ ] Build passes with no warnings
- [ ] All tests pass (529 integration tests)

---

## Architecture: Unified GPU Buffer Pool

```
UnifiedGpuBufferPool (NEW)
├── device: Arc<Mutex<GpuDevice>>
├── memory_pools: HashMap<GpuMemoryPoolId, Arc<Mutex<dyn GpuMemoryPool>>>
├── buffer_cache: LruCache<BufferSpec, Arc<GpuBuffer>>
├── capacity_tracking
│   ├── max_batch_size
│   ├── max_seq_length
│   └── max_embedding_dim
└── GpuComponent trait
    ├── attach_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>)
    ├── gpu_ready(&self) -> Result<()>
    ├── enable_gpu_auto_detect() -> Result<()>
    └── forward_gpu(...) -> Result<Array2<f32>>
```

### Migration Path
```
Phase 5.4 (Current)
├── SharedComponentGpuManager (deprecated)
├── GpuSharedOpsContext (deprecated)
└── Individual component GPU managers

Phase 5.5 (This)
├── UnifiedGpuBufferPool (NEW)
├── GpuComponent trait (NEW)
├── SharedComponentGpuManager (deprecated, wrapper around Unified)
└── GpuSharedOpsContext (deprecated, wrapper around Unified)

Phase 6 (Future)
├── UnifiedGpuBufferPool (STABLE)
├── GpuComponent trait (STABLE)
└── Remove deprecated managers
```

---

## Implementation Tasks

### Task 1: Core Infrastructure (PRIORITY 1)

#### 1.1 Create UnifiedGpuBufferPool
**File**: `src/domain/compute/unified_gpu_buffer_pool.rs`

**Responsibilities**:
- Centralized GPU device management
- Buffer allocation & caching with power-of-2 sizing
- Memory pool coordination
- Capacity tracking for batch_size, seq_len, embed_dim

**Key Methods**:
```rust
pub fn new(device: GpuDevice) -> Self
pub fn auto_detect() -> Result<Self>  // Strict: errors if no GPU
pub fn allocate_buffer(&mut self, spec: BufferSpec) -> Result<Arc<GpuBuffer>>
pub fn get_or_allocate(&mut self, spec: BufferSpec) -> Result<Arc<GpuBuffer>>
pub fn update_capacity(&mut self, batch_size: usize, seq_len: usize, embed_dim: usize)
```

#### 1.2 Create GpuComponent Trait
**File**: `src/domain/compute/gpu_component.rs`

**Requirements**:
```rust
pub trait GpuComponent: Send + Sync {
    fn attach_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) -> Result<()>;
    fn detach_gpu_device(&mut self);
    fn gpu_ready(&self) -> Result<()>;
    fn enable_gpu_auto_detect(&mut self) -> Result<()>;
}
```

**Implementations**:
- SharedAttentionContext
- SharedFeedforward
- TemporalMixingLayer
- PolyAttention

---

### Task 2: Shared Component GPU Variants (PRIORITY 1)

#### 2.1 DiffusionBlock GPU Forward
**File**: `src/domain/layers/diffusion/block.rs`

**Status**: CPU-only, needs GPU path

**Implementation**:
- Add `forward_gpu()` method
- GPU buffer management via `UnifiedGpuBufferPool`
- Diffusion forward: Input -> Embedding -> Denoising -> Output

#### 2.2 SSM GPU Kernels (Mamba/RG-LRU)
**File**: `src/domain/layers/ssm/components/` (new GPU kernel files)

**Status**: Placeholder kernels only

**Implementation**:
- Replace placeholder `temporal_processing_gpu.rs` implementations
- Implement actual WGSL recurrent scan for:
  - Mamba scan operation
  - RG-LRU sequence transformation
- Use selective scan algorithm for efficiency

#### 2.3 TransformerBlock Full GPU Path
**File**: `src/domain/layers/transformer/block.rs`

**Status**: Partial GPU support

**Verification**:
- Full forward chain: Attention GPU -> Temporal GPU -> Feedforward GPU
- Skip connections on GPU
- Post-processing on GPU
- No CPU/GPU transfers in middle of forward pass

---

### Task 3: Cleanup & Deprecation (PRIORITY 2)

#### 3.1 Deprecate SharedComponentGpuManager
**File**: `src/domain/layers/components/shared_gpu_manager.rs`

**Action**:
- Add deprecation warnings
- Create wrapper methods calling `UnifiedGpuBufferPool`
- Document migration path

#### 3.2 Deprecate GpuSharedOpsContext
**File**: `src/domain/layers/components/gpu_shared_ops.rs`

**Action**:
- Add deprecation warnings
- Reimplement using `UnifiedGpuBufferPool`

#### 3.3 Fix ForwardContext Fields
**File**: `src/application/training/forward_context.rs`

**Issues to fix**:
- Missing `low_rank_query_gate` field in initializers
- Ensure all GPU-enabled components have context support

---

### Task 4: PolyAttention Cleanup (PRIORITY 2)

#### 4.1 Resolve Kernel Implementation
**File**: `src/domain/attention/poly_attention.rs`

**Status**: Orphaned code, syntax errors

**Tasks**:
- Complete or remove stub kernel implementations
- Verify `forward_gpu()` path is sound
- Fix any compilation errors

#### 4.2 GPU Fused Kernels
**File**: `src/domain/attention/kernels/poly_attention_fused.wgsl` (new)

**Implementation**:
- Fused PolyAttention kernel combining:
  - Content scoring
  - Position scoring
  - Gate activation
  - Output projection

---

## Testing Strategy

### Unit Tests
- `test_unified_gpu_buffer_pool_*` - Buffer management
- `test_gpu_component_trait_*` - Component interface
- `test_diffusion_block_gpu_forward` - DiffusionBlock GPU
- `test_ssm_gpu_kernels_*` - Mamba/RG-LRU GPU
- `test_transformer_block_full_gpu_chain` - TransformerBlock end-to-end

### Integration Tests
- Full forward pass through TransformerBlock (GPU)
- Full forward pass through DiffusionBlock (GPU)
- Full forward pass through Mamba2 (GPU)
- Backward pass (training) on GPU
- Mixed batch processing (multiple sequences)

### Validation
- Numerical accuracy: GPU vs CPU (ε ≤ 1e-4)
- Performance: GPU speedup vs CPU (target: 2-10x for large batches)
- Memory usage: Tracked via `UnifiedGpuBufferPool::memory_stats()`
- No buffer leaks

---

## Strict No-Fallback Design

### Enforcement
```rust
// BAD: Silent fallback
let result = gpu_forward(input).unwrap_or_else(|_| cpu_forward(input));

// GOOD: Explicit error
let result = gpu_forward(input)?;  // Returns error, doesn't hide it
```

### Error Messages
```
"GPU forward requires device attached. Call enable_gpu_auto_detect() first"
"Automatic GPU detection failed: no supported GPU backend detected"
"CUDA backend requires cudarc feature. Compile with --features gpu-cuda"
"GPU out of memory for buffer size: 1GB (available: 512MB)"
```

### Verification Checklist
- [ ] No `.unwrap_or(cpu_method())` patterns
- [ ] All GPU methods return `Result<T>`
- [ ] No default CPU implementations in GPU traits
- [ ] Errors are descriptive and actionable

---

## Build & Test Commands

### Quick Check (no-GPU)
```bash
cargo check --lib
cargo build --lib
```

### CPU Tests
```bash
cargo test --lib
cargo test --test transformer_block_verification
```

### GPU Tests (if GPU available)
```bash
cargo build --lib --features gpu-wgpu
cargo test --lib --features gpu-wgpu
```

### Coverage
```bash
cargo test --lib 2>&1 | grep "test result"
```

---

## Files to Create/Modify

### Create (NEW)
- `src/domain/compute/unified_gpu_buffer_pool.rs`
- `src/domain/compute/gpu_component.rs`
- `src/domain/layers/diffusion/block_gpu.rs`
- `src/domain/layers/ssm/components/gpu_mamba_kernel.wgsl`
- `src/domain/layers/ssm/components/gpu_rg_lru_kernel.wgsl`
- `src/domain/attention/kernels/poly_attention_fused.wgsl`

### Modify
- `src/domain/compute/mod.rs` - Export Unified* modules
- `src/domain/layers/components/mod.rs` - Add deprecation docs
- `src/domain/layers/diffusion/block.rs` - Add GPU forward
- `src/domain/layers/ssm/components/mod.rs` - Add GPU kernels
- `src/domain/layers/transformer/block.rs` - Verify full GPU chain
- `src/domain/attention/poly_attention.rs` - Fix orphaned code
- `src/application/training/forward_context.rs` - Add missing fields

---

## Consolidation Priority Matrix

| Task | Impact | Effort | Dependencies | Priority |
|------|--------|--------|--------------|----------|
| UnifiedGpuBufferPool | HIGH | HIGH | None | 1 |
| GpuComponent Trait | HIGH | MEDIUM | UnifiedPool | 1 |
| DiffusionBlock GPU | MEDIUM | MEDIUM | GpuComponent | 1 |
| SSM GPU Kernels | MEDIUM | HIGH | GpuComponent | 1 |
| TransformerBlock Verify | HIGH | LOW | GpuComponent | 1 |
| PolyAttention Cleanup | MEDIUM | MEDIUM | GpuComponent | 2 |
| Deprecation Warnings | LOW | LOW | All GPU | 2 |
| Performance Tuning | MEDIUM | HIGH | All complete | 3 |

---

## Performance Targets

### Memory Efficiency
- Buffer reuse rate: >80% (LRU cache hits)
- Power-of-2 alignment efficiency: >95% utilization
- No fragmentation after 100+ allocations

### Compute Performance
- Single-batch latency: match CPU within 10%
- Multi-batch speedup (batch=32): 2-5x vs CPU
- Throughput: >100 samples/sec on modern GPU

### Resource Management
- Peak GPU memory: <available * 0.9
- CPU ↔ GPU transfers minimized (<1% of compute time)
- No memory leaks (validated with heap profiler)

---

## Success Metrics

- [x] Build passes (cargo check)
- [x] All tests pass (529+)
- [ ] UnifiedGpuBufferPool tested (10+ unit tests)
- [ ] GpuComponent trait implemented on 5+ components
- [ ] DiffusionBlock GPU end-to-end verified
- [ ] SSM GPU kernels functional
- [ ] TransformerBlock full GPU chain working
- [ ] Zero deprecation warnings → Phase 6
- [ ] Benchmarks show 2-5x speedup on GPU

---

## References & Links

- **Thread**: @T-019c5ef0-36a1-70be-a528-03e1253f1542
- **Previous Summary**: CONSOLIDATION_GPU_BACKEND_SESSION_SUMMARY.md
- **GPU Status**: GPU_BACKEND_IMPLEMENTATION_STATUS.md
- **Shared Components**: src/domain/layers/components/
- **GPU Compute**: src/domain/compute/
- **Build Guide**: AGENTS.md (Build & Test section)
