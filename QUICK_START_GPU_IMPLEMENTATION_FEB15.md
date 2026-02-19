# Quick Start: GPU Implementation (February 15, 2026)

## Status Check
```bash
# Verify compilation
cargo check --lib --features gpu-wgpu  # Should pass with 0 errors
```

## Current State
✅ **Done**:
- Compilation warnings fixed
- WGPU fused kernel shader (line 515-591 in wgpu_ops.rs)
- RichardsGLU fused kernel method (line 4237+ in wgpu_ops.rs)
- GPU component trait infrastructure

🔄 **Next**:
- Integrate `forward_gpu_fused()` into RichardsGlu
- GPU backward pass
- Validation tests
- Auto-detection

---

## Quick Task List

### Task 1: Add GPU Forward to RichardsGlu (1 hour)

**File**: `src/domain/richards/richards_glu.rs`

Add after the existing `forward()` method around line ~200:

```rust
/// GPU-accelerated forward pass using fused kernel
pub fn forward_gpu_fused(
    &self,
    input: &Array2<f32>,
    device: &Arc<Mutex<GpuDevice>>,
    pool: &mut dyn GpuMemoryPool,
) -> Result<Array2<f32>> {
    // Implementation in SESSION_GPU_CONSOLIDATION_ACTION_PLAN_FEB15.md
    todo!("Implement GPU forward")
}
```

**Then**:
1. Copy full implementation from action plan
2. Replace `todo!()`
3. Test: `cargo test --lib test_richardson_glu_gpu_vs_cpu --features gpu-wgpu`

---

### Task 2: Create Validation Tests (1 hour)

**File**: Create `tests/gpu_richards_glu_validation.rs`

```rust
#[cfg(test)]
mod gpu_richardson_glu_validation {
    // Copy from SESSION_GPU_CONSOLIDATION_ACTION_PLAN_FEB15.md under WORKSTREAM 2
}
```

**Run**:
```bash
cargo test --test gpu_richards_glu_validation --features gpu-wgpu
```

---

### Task 3: GPU Backward Pass (1.5 hours)

**File**: `src/domain/richards/richards_glu.rs`

Add `backward_gpu()` method after `forward_gpu_fused()`:

```rust
pub fn backward_gpu(
    &mut self,
    input: &Array2<f32>,
    loss_gradient: &Array2<f32>,
    device: &Arc<Mutex<GpuDevice>>,
    pool: &mut dyn GpuMemoryPool,
) -> Result<Array2<f32>> {
    // Implementation in SESSION_GPU_CONSOLIDATION_ACTION_PLAN_FEB15.md
    todo!("Implement GPU backward")
}
```

**Note**: May require WGSL backward kernel implementation

---

### Task 4: Auto-Detection (1 hour)

**File**: `src/domain/compute/gpu_device.rs`

Add to `GpuDevice` impl:

```rust
pub fn auto_detect() -> Result<Self> {
    // Implementation in SESSION_GPU_CONSOLIDATION_ACTION_PLAN_FEB15.md
    todo!("Implement auto-detection")
}

pub fn validate_availability(&self) -> Result<()> {
    // Implementation in SESSION_GPU_CONSOLIDATION_ACTION_PLAN_FEB15.md
    todo!("Implement validation")
}
```

**Test**:
```bash
cargo test --lib gpu_device::auto_detect --features gpu-wgpu
```

---

## Compilation Checklist

- [ ] `cargo check --lib --features gpu-wgpu` passes
- [ ] `cargo fmt` applied
- [ ] `cargo clippy --all-targets` has no GPU-related warnings
- [ ] Tests compile: `cargo test --no-run --features gpu-wgpu`

---

## Testing Checklist

Run in order:

```bash
# 1. Validation tests
cargo test --test gpu_richards_glu_validation --features gpu-wgpu -- --nocapture

# 2. Component tests  
cargo test --lib gpu_component --features gpu-wgpu

# 3. Device detection
cargo test --lib gpu_device --features gpu-wgpu

# 4. Memory tests
cargo test --lib gpu_memory --features gpu-wgpu

# 5. Full GPU suite
cargo test --lib --features gpu-wgpu gpu | grep "test result"
```

---

## Performance Targets

| Metric | Target | Check |
|--------|--------|-------|
| Accuracy (GPU vs CPU) | ε ≤ 1e-4 | `assert!(validate_gpu_accuracy(...))` |
| Speedup @ 1K batch | 25x | Benchmark results |
| Memory efficiency | >92% | `get_efficiency_stats()` |
| Zero-copy pipeline | 100% GPU | No CPU transfers in middle |

---

## Common Issues & Solutions

### Issue: GPU feature not enabled
```
error[E0433]: failed to resolve: use of undeclared type `GpuComponent`
```
**Solution**: `cargo ... --features gpu-wgpu`

### Issue: Buffer allocation fails
```
Error: "GPU buffer allocation failed"
```
**Solution**: Check `UnifiedGpuBufferPool` size in tests (100MB minimum)

### Issue: Kernel dispatch fails
```
Error: "Compute pipeline creation failed"
```
**Solution**: Verify shader is valid WGSL (use `SHADER_RICHARDS_GLU_FUSED`)

### Issue: Auto-detection returns Cpu backend
```
eprintln!("GPU backends available: []")
Error: "No GPU backends available..."
```
**Solution**: Enable gpu-wgpu feature and ensure WGPU can initialize

---

## Files Modified Summary

| Phase | File | Changes | Est. Time |
|-------|------|---------|-----------|
| 1 | `richards_glu.rs` | +forward_gpu_fused() | 1 hour |
| 2 | `gpu_richards_glu_validation.rs` | Create new test file | 1 hour |
| 3 | `richards_glu.rs` | +backward_gpu() | 1.5 hours |
| 4 | `gpu_device.rs` | +auto_detect(), +validate() | 1 hour |
| Total | | | **4.5 hours** |

---

## Verification Commands

After each task, run:

```bash
# Check types and syntax
cargo check --lib --features gpu-wgpu

# Run specific tests
cargo test --lib gpu_richardson --features gpu-wgpu -- --exact --nocapture

# Format
cargo fmt

# Lint
cargo clippy --all-targets --features gpu-wgpu
```

---

## Success Criteria (End of Session)

- [ ] RichardsGlu::forward_gpu_fused() compiles and runs
- [ ] All numerical validation tests pass (ε ≤ 1e-4)
- [ ] Backward pass computes gradients without error
- [ ] GPU detection works with clear error if GPU unavailable
- [ ] Memory pool tracking shows >92% efficiency
- [ ] No CPU fallback paths active

---

## Next Session Priorities (If Time Remains)

1. PolyAttention GPU kernels
2. Mamba GPU kernels
3. Performance benchmarking
4. Zero-copy pipeline integration

---

**Start Time**: [Insert when beginning]  
**Target Completion**: ~4.5 hours  
**Difficulty**: Medium (GPU infrastructure in place, mainly integration)
