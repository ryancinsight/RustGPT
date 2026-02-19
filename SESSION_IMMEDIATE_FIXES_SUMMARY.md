# Immediate Build Fixes Required

## Status Summary
- **Build Errors**: 13+ compilation errors
- **Previous Progress**: Fixed anyhow import, partially fixed ForwardContext (needs low_rank_query_gate in all 4 initializers)
- **Key Blockers**: Missing MoHGating.low_rank_query_gate field, PolyAttention GPU field issues

## Critical Fixes Needed (In Priority Order)

### 1. Add `low_rank_query_gate` field to MoHGating ✅ DONE
- File: `src/domain/mixtures/moh_gating.rs`
- Line 404+: Added field `pub low_rank_query_gate: RichardsCurve`
- Line 465: Initialize in new() with `RichardsCurve::sigmoid(true)`

### 2. Add `low_rank_query_gate` to all 4 ForwardContext initializers in PolyAttention
- File: `src/domain/attention/poly_attention.rs`
- Lines: 1353, 1422, 1510, 1582
- Fix: Add `low_rank_query_gate: &self.moh.low_rank_query_gate,` to each

### 3. Add missing 6th argument to PolyAttentionParamInfo::new
- File: `src/domain/attention/poly_attention.rs`
- Line 2504-2511
- Fix: Add `low_rank_gate_params` calculation and parameter
- Value: Calculate from `self.moh.low_rank_query_gate` or estimate as 15

### 4. Remove GPU field references from poly_attention_gpu.rs
- File: `src/domain/attention/poly_attention_gpu.rs`
- Lines: 22, 28, 33, 40
- Issue: `self.gpu_device` field doesn't exist on PolyAttention
- Fix: Remove entire gpu_device field usage or relocate to GpuComponent trait

### 5. Implement `forward_gpu` for PolyAttention (or remove call)
- File: `src/domain/layers/components/common.rs`
- Line 291: TemporalMixing variant calls `layer.forward_gpu(input)`
- Issue: PolyAttention doesn't impl GpuTemporalOps
- Fix Option A: Comment out GPU call for PolyAttention (Phase 5.5 stub)
- Fix Option B: Impl GpuTemporalOps for PolyAttention with CPU fallback

## Reapplication of Previous Fixes

Need to reapply these from earlier attempts:
1. ✅ MoHGating: Added low_rank_query_gate field
2. ❌ PolyAttention: ForwardContext fixes (4 locations)
3. ❌ PolyAttention: PolyAttentionParamInfo argument
4. ❌ Remove GpuMatrixOps unused import
5. ❌ PolyAttention GPU stub functions

## Next Steps

1. Apply all 4 ForwardContext fixes systematically
2. Add low_rank_gate_params calculation
3. Remove or stub GPU device field usage  
4. Comment out forward_gpu call for PolyAttention
5. Run `cargo check` to verify build passes

## Test Plan

1. ✅ `cargo check` - should pass with 0 errors
2. ✅ `cargo test --lib` - verify 529+ tests pass
3. ✅ GPU device auto-detection test (strict no-fallback)
4. ✅ Verify CPU forward paths work correctly
