# Phase 5.6 Consolidation & GPU Backend Implementation Plan
**Date**: February 15, 2026  
**Focus**: Consolidation of shared components (Attention Context, Feedforward, Temporal) with GPU backend variants  
**Strategy**: Automatic GPU detection with strict no-fallback semantics  
**Status**: Ready for execution

---

## 1. Current State & Blockers

### ✅ Completed Foundations
- `UnifiedGpuBackend` infrastructure established
- `UnifiedGpuKernels` dispatcher created
- `UnifiedBufferPool` for cross-architecture memory sharing
- `GpuComponent` trait pattern implemented
- Zero compiler warnings achieved

### 🔧 Infrastructure Complete
- `attention_context_gpu.rs` - GPU helpers for attention context modulation
- `feedforward_gpu.rs` - GPU helpers for RichardsGLU and MoE
- `temporal_processing_gpu.rs` - GPU helpers for attention/SSM/RG-LRU
- `fused_kernels_module.rs` - Kernel stubs awaiting implementation

### ⚠️ Critical Issue Fixed
- **File**: `attention_context_gpu.rs` line 97
- **Issue**: Missing `use ndarray::linalg::general_mat_mul;` import
- **Impact**: Prevents compilation of covariance computation
- **Status**: FIXED ✅

---

## 2. Consolidation Architecture Overview

```
SharedComponents (Phase 5.6)
├── AttentionContext
│   ├── CPU Path: matrix multiplication + EMA update
│   └── GPU Path: unified backend dispatch
├── Feedforward  
│   ├── RichardsGLU (with Richards curve activation)
│   └── MixtureOfExperts (if implemented)
│       ├── CPU Path: standard feed-forward
│       └── GPU Path: fused kernel dispatch
└── TemporalProcessing
    ├── Transformer Attention
    ├── Mamba SSM
    └── RG-LRU
        ├── CPU Path: recurrent processing
        └── GPU Path: unified backend dispatch
```

### GPU Dispatch Flow
```
Forward Pass
├── Check GPU availability (strict no-fallback)
├── Transfer input once to GPU
├── Execute fused kernels:
│   ├── Linear projections (Q/K/V or W1/W2)
│   ├── Activation/Gating
│   └── Output projection
└── Transfer result back to CPU (zero-copy for intermediates)
```

---

## 3. Immediate Actions (Priority Order)

### Phase 3.1: Wire SharedAttentionContext GPU

**File**: `src/domain/layers/components/attention_context.rs`

#### Tasks
1. Add GPU device field:
   ```rust
   #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
   gpu_device: Option<Arc<Mutex<GpuDevice>>>,
   ```

2. Implement `GpuComponent` trait using macro:
   ```rust
   impl_gpu_component!(SharedAttentionContext);
   ```

3. Update `apply_incoming_context()` to dispatch:
   ```rust
   pub fn apply_incoming_context(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
       #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
       {
           if let Some(device) = &self.gpu_device {
               let mut backend = UnifiedGpuBackend::from_device(device.clone());
               return self.apply_incoming_context_gpu(input, &mut backend);
           }
       }
       // CPU path (existing code)
       self.apply_incoming_context_cpu(input)
   }
   ```

4. **Compile & Test**:
   ```bash
   cargo check --lib
   cargo test --lib attention_context -- --test-threads=1
   ```

---

### Phase 3.2: Wire SharedFeedforward GPU

**File**: `src/domain/layers/components/feedforward.rs`

#### Tasks
1. Add GPU device field to `SharedFeedforward`:
   ```rust
   #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
   gpu_device: Option<Arc<Mutex<GpuDevice>>>,
   ```

2. Implement `GpuComponent` + `GpuFeedforwardOps` traits.

3. Update `forward()` with GPU dispatch:
   ```rust
   pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
       #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
       {
           if self.is_gpu_ready() {
               if let Ok(mut backend) = UnifiedGpuBackend::auto_detect() {
                   return self.forward_gpu_with_backend(input, &mut backend);
               }
           }
       }
       // CPU path (existing code)
       self.forward_cpu(input)
   }
   ```

4. **Compile & Test**:
   ```bash
   cargo check --lib
   cargo test --lib feedforward -- --test-threads=1
   ```

---

### Phase 3.3: Wire SharedTemporalProcessing GPU

**File**: `src/domain/layers/components/temporal_processing.rs`

#### Tasks
1. Add GPU device field to `TemporalMixingLayer`:
   ```rust
   #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
   gpu_device: Option<Arc<Mutex<GpuDevice>>>,
   ```

2. Implement `GpuComponent` + `GpuTemporalOps` traits.

3. Update `forward()` with GPU dispatch based on `TemporalMixingType`:
   ```rust
   pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
       #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
       {
           if self.is_gpu_ready() {
               if let Ok(mut backend) = UnifiedGpuBackend::auto_detect() {
                   return self.forward_gpu_with_backend(input, &mut backend);
               }
           }
       }
       // CPU path (dispatch to attention/mamba/rg_lru)
       match self.mixing_type {
           TemporalMixingType::Attention => self.forward_attention_cpu(input),
           TemporalMixingType::Mamba => self.forward_mamba_cpu(input),
           TemporalMixingType::RgLru => self.forward_rg_lru_cpu(input),
       }
   }
   ```

4. **Compile & Test**:
   ```bash
   cargo check --lib
   cargo test --lib temporal_processing -- --test-threads=1
   ```

---

### Phase 3.4: Integration Testing

**File**: `tests/gpu_shared_components_integration.rs` (create new)

#### Tests to Add
1. **GPU Auto-Detection**:
   - Verify strict no-fallback (error on CPU-only systems)
   - Verify detection priority (CUDA > Metal > Vulkan > WGPU)

2. **AttentionContext GPU**:
   - Compare GPU output vs. CPU (tolerance: 1e-4)
   - Test with varying batch sizes (32, 128, 512, 1024)

3. **Feedforward GPU**:
   - RichardsGLU forward pass on GPU
   - Compare gradient computation vs. CPU

4. **TemporalProcessing GPU**:
   - All three variants (Attention, Mamba, RG-LRU)
   - Zero-copy verification (no CPU<->GPU transfers for intermediates)

#### Compile & Run Full Suite
```bash
cargo test --test gpu_shared_components_integration --features gpu-wgpu
cargo test --test gpu_shared_components_integration --features gpu-cuda
```

---

## 4. Fused Kernel Implementation Roadmap

### Kernel Priority Order

#### 1️⃣ **Attention Context Kernel** (Lowest Hanging Fruit)
- **Operation**: `input @ context_strength`
- **Kernel Reduction**: Already single pass (matrix multiply)
- **Target**: Forward + backward in unified kernel
- **File**: `unified_gpu_kernels.rs`

#### 2️⃣ **RichardsGLU Fused Kernel**
- **Current**: 5 GPU launches (W1, W2 projections + Richards activation + W_out)
- **Target**: 2 passes
  - **Pass 1**: W1 proj → Richards curve → W2 proj → Gating
  - **Pass 2**: Output projection
- **File**: `fused_kernels_module.rs` → `richards_glu_fused()`

#### 3️⃣ **PolyAttention Fused Kernel**
- **Current**: 3+ launches (projections + scoring + softmax)
- **Target**: 1 pass (Q/K/V projections → scoring → softmax)
- **File**: `fused_kernels_module.rs` → `poly_attention_fused()`

#### 4️⃣ **Mamba Scan Kernel**
- **Current**: Per-step recurrence (sequential)
- **Target**: Vectorized selective scan with state compression
- **File**: `fused_kernels_module.rs` → `mamba_scan_kernel()`

---

## 5. GPU Backend Detection & Error Handling

### No-Fallback Policy Implementation

```rust
// Pattern: Strict GPU detection
pub fn auto_detect() -> Result<UnifiedGpuBackend> {
    // Try in priority order
    if let Ok(device) = GpuDevice::detect_cuda() {
        return Ok(UnifiedGpuBackend::from_device(device));
    }
    if let Ok(device) = GpuDevice::detect_metal() {
        return Ok(UnifiedGpuBackend::from_device(device));
    }
    if let Ok(device) = GpuDevice::detect_vulkan() {
        return Ok(UnifiedGpuBackend::from_device(device));
    }
    if let Ok(device) = GpuDevice::detect_wgpu() {
        return Ok(UnifiedGpuBackend::from_device(device));
    }
    
    // No GPU found - error instead of fallback
    Err(ModelError::Backend {
        message: "No GPU detected. Available: CUDA, Metal, Vulkan, WGPU. \
                  Remove GPU dispatch code or install a compatible GPU runtime.".to_string(),
    })
}
```

### Error Messages
- Clear GPU availability status
- Detection priority information
- Remediation steps (e.g., "Install CUDA Toolkit 12.4+")

---

## 6. Memory Efficiency & Zero-Copy Strategy

### Buffer Lifetime Management

```
GPU Forward Pass
├── [1] Upload: input → GPU (once)
├── [2] Compute: all kernels on GPU
│   ├── Attention/FFN projections
│   ├── Activation functions
│   └── Output projections
├── [3] Download: output → CPU (once)
└── [4] NO intermediate CPU transfers
```

### Buffer Pool Optimization
- Pre-allocate power-of-2 sized buffers (eliminates fragmentation)
- Reuse across consecutive operations
- Reset pool on GPU device switch

```rust
// Usage in kernels
let buffer = backend.buffer_pool.allocate(size)?;
// Compute...
backend.buffer_pool.recycle(buffer);
```

---

## 7. Performance Targets (Phase 5.6+)

| Component | Operation | CPU Ref | GPU Target | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **AttentionContext** | 1K batch, 64D | 15ms | 0.5ms | **30x** |
| **RichardsGLU** | 1K batch, 4096D→16384D | 50ms | 2ms | **25x** |
| **PolyAttention** | 512 batch, 64D, 8 heads | 30ms | 1ms | **30x** |
| **Mamba Scan** | 512 seq, 64D | 40ms | 2ms | **20x** |

---

## 8. Validation & Testing Strategy

### Correctness Validation
```bash
# Test CPU vs GPU numerical accuracy (tolerance: ε ≤ 1e-4)
cargo test test_gpu_accuracy --lib

# Test feature flag combinations
cargo test --lib --features gpu-wgpu
cargo test --lib --features gpu-cuda
cargo test --lib --features gpu-metal

# Test no-fallback error handling
cargo test test_gpu_no_fallback --lib
```

### Performance Benchmarking
```bash
cargo bench --bench gpu_shared_components_phase56
cargo bench --bench transformer_block_gpu
```

---

## 9. Compiler & Format Compliance

### Pre-Commit Checklist
```bash
# Format check (AGENTS.md: rustfmt with edition 2024)
cargo fmt -- --check

# Linting
cargo clippy --all-targets

# Full compilation
cargo check --lib
cargo build --release --features gpu-all

# Run test suite
cargo test --lib
```

---

## 10. File Modifications Summary

### Files to Modify (Wiring GPU Dispatch)
1. ✅ **attention_context_gpu.rs** - Fixed import (line 13)
2. **attention_context.rs** - Add GPU device field + dispatch
3. **feedforward.rs** - Add GPU device field + dispatch  
4. **temporal_processing.rs** - Add GPU device field + dispatch

### Files to Create (Integration Tests)
1. **tests/gpu_shared_components_integration.rs** - Full GPU integration suite

### Files to Enhance (Kernel Implementation)
1. **unified_gpu_kernels.rs** - Implement `forward_attention_context()`
2. **fused_kernels_module.rs** - Implement fused kernels (stubs → full impl)

---

## 11. Next Session Quick Start

### Build & Verify Current State
```bash
cd d:/RustGPT
cargo build --release --features gpu-wgpu
```

### Implementation Order
1. **Start**: `attention_context.rs` wiring (simplest)
2. **Continue**: `feedforward.rs` wiring
3. **Finish**: `temporal_processing.rs` wiring
4. **Test**: Run integration tests with GPU features

### Expected Outcomes
- Zero compiler warnings ✅
- All GPU dispatch paths active
- Auto-detection working (strict no-fallback)
- Integration tests passing (if GPU available)

---

## 12. Risk Mitigation

| Risk | Mitigation |
| :--- | :--- |
| GPU not available in CI | Tests skip gracefully; no fallback errors |
| Feature flag mismatch | Compile-time cfg guards prevent issues |
| Memory leaks in CUDA | UnifiedBufferPool manages lifecycle |
| Numerical divergence | Tolerance tests (ε ≤ 1e-4) validate accuracy |
| Kernel crash on edge cases | Dimension validation before GPU dispatch |

---

**Status**: Ready for execution. All blockers cleared. Proceeding with Phase 3.1 wiring.
