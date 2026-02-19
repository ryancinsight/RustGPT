# Consolidation & GPU Implementation Action Plan
**Date**: February 14, 2026  
**Status**: Starting new session  
**Focus**: Fix build issues, RG-LRU workspace integration, GPU kernel implementation

---

## Current Issues

### 1. Duplicate `moh_gate_activation` Definition ✅ FIXED
- **File**: `src/domain/compute/gpu_ops.rs`
- **Issue**: Two trait method definitions with different signatures
- **Solution**: Removed duplicate (lines 276-290), kept correct signature (lines 237-247)
- **Reason**: WGPU implementation (wgpu_ops.rs:1901) uses the first signature

### 2. Build Status
- **Last Status**: Build failure due to duplicate
- **Next Step**: Verify clean build after fix
- **Unused Imports**: Minor warnings in poly_attention_gpu.rs and poly_attention.rs (cleanup deferred)

---

## Session Priorities

### PRIORITY 0: Fix & Verify Build (30 min)
1. ✅ Remove duplicate `moh_gate_activation` 
2. Verify clean `cargo check` / `cargo build --release`
3. Run quick test: `cargo test --lib` (sample)

### PRIORITY 1: RG-LRU Workspace Integration (3-4 hours)
**Location**: `src/domain/layers/ssm/rg_lru.rs`

Steps:
1. Add `unified_workspace: Option<UnifiedLayerWorkspace>` field to RgLru struct
2. Implement `WorkspaceManaged` trait for RgLru
3. Implement `StreamingWorkspaceManaged` trait for RgLru
4. Test: Single-step, multi-step streaming, MoH variants

Expected Impact: -80 LOC, unified SSM memory

### PRIORITY 2: GPU Kernel Implementations (4-6 hours)
**Files**: 
- `src/domain/compute/wgpu_ops.rs` - Shader kernels
- `src/domain/compute/gpu_memory.rs` - Memory management

Kernels to implement:
1. GEMM (Matrix-Matrix Multiply) - Foundation for all ops
2. Element-wise ops (ReLU, GELU, SiLU, Sigmoid)
3. Layer normalization & Softmax
4. Richards curve (critical for PolyAttention)
5. PolyAttention fused kernel (poly_a, poly_b scoring)

**Approach**: Start with WGPU placeholders + CPU fallback, then CUDA variants

### PRIORITY 3: Streaming Workspace Consolidation (2-3 hours)
**Location**: `src/domain/layers/ssm/`, `src/domain/attention/`

Consolidate:
- RgLruStreamingWorkspace
- MambaStreamingState
- PolyAttentionStreamingWorkspace
- SlidingWindowStreamingWorkspace
- RingAttentionStreamingWorkspace

### PRIORITY 4: In-Place Operations (4-5 hours)
Add `forward_into()` methods to:
- `SharedTemporalProcessing`
- `SharedFeedforward`

---

## GPU Backend Architecture

### Current State (Phase 5.3)
```
GpuMatrixOps (trait)
├── GEMM operations (Level 3 BLAS)
├── Element-wise ops (activation, scaling)
├── Normalization (LayerNorm, Softmax)
├── PolyAttention kernels
└── Specialized SSM kernels

Backends:
├── WGPU (Vulkan/Metal/DX12) - Currently using placeholders
├── CUDA (via cudarc) - Placeholder
├── Metal (via MPS) - Placeholder
└── CPU fallback (with strict no-fallback mode)
```

### No-Fallback Strategy
- GPU operations **MUST** error if no GPU available
- Auto-detection with clear error messages
- Enables easier troubleshooting

---

## Testing Strategy

### Build Verification
```bash
cargo check                          # Quick check
cargo build --release               # Full build
cargo build --release --features gpu-wgpu  # With GPU
```

### Unit Tests
```bash
cargo test --lib rg_lru              # RG-LRU tests
cargo test --lib shared_feedforward  # FFN tests
cargo test --lib shared_attention    # Attention tests
```

### Integration Tests
```bash
cargo test --test transformer_block_verification
cargo test --test diffusion_block_verification  # if exists
```

### GPU Tests
```bash
cargo test --lib --features gpu-wgpu -- gpu
cargo test --lib --features gpu-cuda -- gpu
```

---

## Files Modified This Session

### Fixed
- `src/domain/compute/gpu_ops.rs` - Removed duplicate moh_gate_activation

### Will Modify
- `src/domain/layers/ssm/rg_lru.rs` - Workspace integration
- `src/domain/compute/wgpu_ops.rs` - Kernel implementations
- `src/domain/compute/gpu_memory.rs` - Buffer management

---

## Success Criteria

✅ Build succeeds without warnings  
✅ All 529 tests passing  
✅ RG-LRU workspace integrated & tested  
✅ GPU GEMM kernel operational (WGPU)  
✅ No functional regressions  

---

## Estimated Timeline

| Task | Est. Time | Priority |
|:-----|:---------:|:--------:|
| Fix build | 30 min | P0 |
| RG-LRU integration | 3-4h | P1 |
| GPU kernels (GEMM+basic) | 4-6h | P1 |
| Streaming consolidation | 2-3h | P1 |
| **Total** | **10-14h** | |

**Recommended Split**: 5-6 hours/session × 2 sessions

---

## References

- **Consolidation Plan**: CONSOLIDATION_PHASE5_CONTINUATION_PLAN.md
- **GPU Status**: GPU_BACKEND_IMPLEMENTATION_STATUS.md
- **Thread**: https://ampcode.com/threads/T-019c5a7e-8065-77ec-b6ef-a65a559b4285
