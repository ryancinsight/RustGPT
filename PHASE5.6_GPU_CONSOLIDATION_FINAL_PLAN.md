# Phase 5.6: GPU Backend Consolidation and Performance Optimization Plan

**Status**: Consolidation Phase Started (Feb 15, 2026)  
**Thread**: @T-019c6417-73e1-747f-98d9-4925a2fc44a5  
**Current Focus**: Cleanup, GPU backend variant implementations, automatic detection with strict no-fallback

## Executive Summary

Phase 5.6 focuses on consolidating GPU backends across diffusion, SSM, and transformer shared components with a strict no-fallback GPU detection policy. This phase eliminates silent CPU fallback and ensures predictable performance characteristics through:

1. **Cleanup Pass** (COMPLETED)
   - Remove unused imports and parameters
   - Enforce strict compilation (no warnings)
   - Add proper feature-gating for GPU-specific code

2. **GPU Backend Variant Implementation** (IN PROGRESS)
   - Implement actual fused kernel placeholders
   - Wire GPU dispatch in shared components
   - Add GPU detection with strict error semantics

3. **Automatic GPU Detection** (IN PROGRESS)
   - Implement `GpuDevice::auto_detect()` with detection priority
   - Error handling for feature flag mismatches
   - No CPU fallback (explicit error instead)

4. **Memory Efficiency Optimization** (PLANNED)
   - Implement power-of-2 buffer sizing
   - Unified buffer pool across architectures
   - Streaming data pipeline (keep on GPU between ops)

5. **Performance Validation** (PLANNED)
   - Benchmark against CPU implementations
   - Profile kernel execution time
   - Memory utilization tracking

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              UnifiedGpuBackend (Entry Point)            │
│  - auto_detect() - strict no-fallback detection         │
│  - forward_attention_context()                          │
│  - forward_feedforward()                                │
│  - forward_temporal()                                   │
│  - Stats/memory tracking                                │
└─────────────────────────────────────────────────────────┘
           ↓                    ↓                    ↓
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │ Attention Ops │  │ Feedforward   │  │ Temporal      │
    │ (GEMM-based)  │  │ (Fused)       │  │ (SSM/Attn)    │
    └───────────────┘  └───────────────┘  └───────────────┘
           ↓                    ↓                    ↓
    ┌─────────────────────────────────────────────────────┐
    │    GPU Device (CUDA/Metal/Vulkan/WGPU)             │
    │  - Automatic backend detection                      │
    │  - Strict error on unavailability                   │
    │  - Matrix operations (GEMM, element-wise)           │
    └─────────────────────────────────────────────────────┘
```

---

## Phase Milestones

### Milestone 1: Cleanup (✅ COMPLETED)
- [x] Remove all unused imports
- [x] Fix all unused parameters with underscore prefix
- [x] Add proper feature-gating to conditional imports
- [x] Zero compiler warnings in `cargo check --lib`

### Milestone 2: GPU Variant Implementation (🔄 IN PROGRESS)

#### 2.1 Fused Kernel Stubs (WIP)
**Target**: Replace TODO placeholders with actual GPU operations

**Files to implement**:
- `fused_kernels_module.rs`
  - `richards_glu_fused::execute()` - Two-pass fused kernel
  - `poly_attention_fused::execute()` - Single-pass attention kernel
  - `mamba_scan_kernel::execute()` - Selective scan kernel
  - `attention_context_ops::{apply_incoming_context, update_outgoing_context}()`

**Implementation strategy**:
```rust
// Phase 5.6.3 Pattern: Use device/pool/ops to dispatch kernels
pub fn execute(
    _device: &Arc<Mutex<GpuDevice>>,  // Will be used in 5.6.3
    _pool: &mut dyn GpuMemoryPool,    // Will be used in 5.6.3
    _ops: &mut dyn GpuMatrixOps,      // Will be used in 5.6.3
    input: &Array2<f32>,
    // ... kernel-specific inputs ...
) -> Result<Array2<f32>> {
    // 5.6.3: Actual GPU kernel dispatch
    // For now: placeholder (return input clone)
    Ok(input.clone())
}
```

#### 2.2 Wire GPU Dispatch in Components (TODO)
**Files to update**:
- `src/domain/layers/components/attention_context.rs`
  - Add `enable_gpu()` method to wire UnifiedGpuBackend
  - Update `apply_incoming_context()` to check GPU availability
  - Update `update_outgoing_context()` to check GPU availability

- `src/domain/layers/components/shared_feedforward.rs`
  - Add `enable_gpu()` method
  - Update `forward()` to dispatch to GPU when available

- `src/domain/layers/components/shared_temporal_processing.rs`
  - Add `enable_gpu()` method
  - Dispatch attention/mamba/rglru to appropriate GPU kernels

**Pattern**:
```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn forward_gpu(&mut self, input: &Array2<f32>, backend: &mut UnifiedGpuBackend) -> Result<Array2<f32>> {
    backend.forward_attention_context(input, &self.context, 1.0)
}
```

#### 2.3 Automatic GPU Detection (READY FOR TESTING)
**Status**: CUDA > Metal > Vulkan > WGPU priority implemented in `GpuDevice::auto_detect()`

**Error handling**:
```
If GPU hardware detected but feature flags missing:
  "Automatic GPU detection found runtime backend(s): [Backend]. 
   This binary was not built with matching GPU feature flags. 
   Enable one of: --features gpu-cuda, gpu-metal, gpu-wgpu."

If no GPU detected:
  "No GPU device available on this system"
```

**Testing approach**:
```bash
# Build with WGPU (should work on all platforms)
cargo check --lib --features gpu-wgpu

# Build without GPU features (should error on GPU detection)
cargo check --lib

# Test auto-detect in tests
#[test]
fn test_gpu_auto_detect_strict() {
    match UnifiedGpuBackend::auto_detect() {
        Ok(backend) => println!("GPU: {}", backend.backend_name()),
        Err(e) => println!("No GPU available: {}", e),
    }
}
```

---

## Milestone 3: Memory Efficiency (PLANNED for 5.6.2)

### 3.1 Unified Buffer Pool

**Implementation**:
```rust
pub struct UnifiedBufferPool {
    // Power-of-2 sized pools to reduce fragmentation
    pools: HashMap<usize, VecDeque<GpuBuffer>>,  // size -> available buffers
}
```

**Target metrics**:
- Memory utilization: >92%
- Fragmentation: <5%
- Buffer reuse count: 10-100x reduction in allocations

### 3.2 Zero-Allocation Forward Pipeline

**Pattern**:
```
Input Upload → GPU Kernel Chain → Output Download
     ↓              ↓                     ↓
   Once        On GPU (no CPU)         Once
```

**Components**:
- Attention → Temporal → Feedforward (all on GPU)
- No intermediate CPU transfers

---

## Milestone 4: Performance Validation (PLANNED)

### 4.1 Benchmarking

| Component | Op | CPU Ref | GPU Target | Speedup |
|-----------|-----|---------|------------|---------|
| RichardsGLU | 1K batch | 50ms | 2ms | 25x |
| PolyAttention | 512 batch | 30ms | 1ms | 30x |
| Mamba Scan | 512 batch | 40ms | 2ms | 20x |
| AttentionContext | 1K batch | 15ms | 0.5ms | 30x |

### 4.2 Profiling and Tracing

**Tools**:
- `NSight` for CUDA profiling
- `XCode Instruments` for Metal profiling
- `WGPU` built-in timing

---

## Current Status by Component

### ✅ Completed
- [x] `UnifiedGpuBackend` trait and core implementation
- [x] `GpuDevice::auto_detect()` with strict no-fallback
- [x] `forward_attention_context()` GPU path
- [x] `forward_feedforward()` GPU path (stub)
- [x] GPU feature-gating and imports cleanup
- [x] No compiler warnings

### 🔄 In Progress
- [ ] Fused kernel implementations (stubs → placeholders → actual)
- [ ] Wire GPU dispatch in SharedAttentionContext
- [ ] Wire GPU dispatch in SharedFeedforward
- [ ] Wire GPU dispatch in SharedTemporalProcessing
- [ ] Testing with `--features gpu-wgpu`

### ⏳ Planned
- [ ] RichardsGLU two-pass kernel (Phase 5.6.3)
- [ ] PolyAttention single-pass kernel (Phase 5.6.3)
- [ ] Mamba selective scan kernel (Phase 5.6.3)
- [ ] Unified buffer pool implementation
- [ ] Zero-copy pipeline
- [ ] Performance benchmarks

---

## Implementation Checklist

### Phase 5.6.1: Core Consolidation (NOW)

```
GPU Backend Infrastructure
  [x] UnifiedGpuBackend struct
  [x] Auto-detect with priority order
  [x] Stats tracking (kernel launches, bytes transferred)
  [x] GpuActivation enum
  [x] GpuTemporalType enum
  [x] Feature gating cleanup

Fused Kernel Stubs
  [x] RichardsGLU params structure
  [x] PolyAttention params structure
  [x] Mamba selective scan params
  [x] Placeholder implementations
  
GPU Component Integration
  [ ] SharedAttentionContext GPU methods
  [ ] SharedFeedforward GPU methods
  [ ] SharedTemporalProcessing GPU methods
  [ ] Test GPU auto-detect
```

### Phase 5.6.2: Kernel Implementation (Next)

```
Actual Kernel Dispatch
  [ ] RichardsGLU two-pass execution
  [ ] PolyAttention single-pass execution
  [ ] Mamba selective scan execution
  [ ] Attention context matrix multiplication

Buffer Pool
  [ ] Power-of-2 sizing implementation
  [ ] Pool segregation by size
  [ ] Lazy allocation
  [ ] Reuse statistics
  
Memory Management
  [ ] Zero-copy forward pass
  [ ] Buffer lifetime management
  [ ] Stream-based pipeline
```

### Phase 5.6.3: Performance Tuning (Next+1)

```
Kernel Fusion
  [ ] Pass 1: RichardsGLU (W1→Richards→W2→Gate)
  [ ] Pass 2: RichardsGLU (W_out)
  [ ] Single-pass PolyAttention
  [ ] Recurrent Mamba kernel

Optimization
  [ ] Warp-level reductions
  [ ] Shared memory utilization
  [ ] Occupancy tuning
  [ ] Global memory coalescing
```

---

## GPU Detection Testing Matrix

### With `--features gpu-wgpu`

```
System         | Expected Behavior
----------------|------------------
NVIDIA GPU     | Detect CUDA (preferred), then fall through to WGPU
AMD GPU        | Detect Vulkan (preferred), then fall through to WGPU
Intel GPU      | Detect Metal or Vulkan, fall through to WGPU
Apple GPU      | Detect Metal, fall through to WGPU
No GPU         | Error: "No GPU device available"
```

### Without GPU features

```
System         | Expected Behavior
----------------|------------------
Any GPU        | Error: "GPU detected but feature flags missing"
No GPU         | Error: "No GPU device available" (OK)
```

---

## Common Issues & Troubleshooting

### Issue: "GPU detected but feature flags missing"
**Cause**: GPU hardware found at runtime, but binary compiled without GPU support.  
**Solution**: 
```bash
cargo build --release --features gpu-wgpu
# or
cargo build --release --features gpu-cuda,gpu-metal
```

### Issue: "No GPU device available"
**Cause**: No GPU detected on system.  
**Solution**: 
- This is expected behavior - the system has no GPU
- Use CPU computation instead (remove GPU code paths)
- Or add GPU hardware to system

### Issue: GPU operations silently fall back to CPU
**Cause**: This should NOT happen - Phase 5.6 enforces strict no-fallback.  
**Solution**: 
- Check error messages carefully
- Verify GPU is available with `nvidia-smi` or `metal-gpu-metrics`
- File bug report if silent fallback occurs

---

## Next Steps (IMMEDIATE)

1. **Wire GPU dispatch in SharedAttentionContext** (30 min)
   - Add `gpu_backend: Option<UnifiedGpuBackend>` field
   - Implement `apply_incoming_context_gpu()` wire-up
   - Test with GPU auto-detect

2. **Test with WGPU feature** (20 min)
   ```bash
   cargo check --lib --features gpu-wgpu
   cargo test --lib --features gpu-wgpu
   ```

3. **Verify strict no-fallback** (15 min)
   - Run GPU tests with GPU available (should work)
   - Run GPU tests without GPU (should error clearly)
   - Verify no CPU fallback occurs

4. **Consolidate remaining components** (1-2 hours)
   - SharedFeedforward GPU wiring
   - SharedTemporalProcessing GPU wiring
   - Component integration tests

5. **Create comprehensive test suite** (1-2 hours)
   - GPU detection tests
   - Feature flag mismatch tests
   - Kernel execution tests
   - Memory stats tests

---

## Documentation & References

- **Strict No-Fallback Policy**: Errors if GPU unavailable, never silently falls back to CPU
- **Detection Priority**: CUDA > Metal > Vulkan > WGPU
- **Feature Flags**: `gpu-cuda`, `gpu-metal`, `gpu-wgpu` (can combine)
- **Thread**: @T-019c6417-73e1-747f-98d9-4925a2fc44a5

