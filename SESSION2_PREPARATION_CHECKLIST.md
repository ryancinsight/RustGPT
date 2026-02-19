# Session 2 Preparation Checklist - WGPU BLAS Implementation
**Target Date**: February 15, 2026  
**Duration**: Full session  
**Goal**: Implement WGPU GEMM and complete Session 1 tasks

---

## Pre-Session Setup (Do 30 min before session starts)

### Code Verification
- [ ] `cd d:\RustGPT && git pull origin main`
- [ ] `cargo test --lib` → Expect 539 passing tests
- [ ] `cargo build --features wgpu` → Should compile successfully
- [ ] `cargo clippy --all-targets` → 0 errors in GPU modules

### Documentation Review
- [ ] Read: `QUICK_START_PHASE5.5.md` - entire file (20 min)
- [ ] Read: `WGPU_BLAS_IMPLEMENTATION_GUIDE.md` sections 1-6 (30 min)
- [ ] Skim: `GPU_CONSOLIDATION_PHASE5.5_ACTION_PLAN.md` sections 1-2 (10 min)

### Environment Check
- [ ] GPU available: Run `nvidia-smi` (NVIDIA) or check Metal (Apple)
- [ ] WGPU backends available: At least one of Vulkan/DX12/Metal works
- [ ] Rust toolchain: `rustc --version` (should be 1.85+)
- [ ] IDE/Editor ready: VS Code with rust-analyzer

---

## Session 1 Deliverables (Must Complete)

### Checkpoint 1.1: Project Structure (Hour 0-0.5)
- [ ] Create `src/domain/compute/wgpu_shaders/` directory
- [ ] Create shader files:
  - [ ] `gemm.wgsl`
  - [ ] `activation.wgsl`
  - [ ] `softmax.wgsl`
  - [ ] `attention.wgsl` (may be stub)
- [ ] Create `src/domain/compute/wgpu_ops_blas.rs`
- [ ] Create `tests/gpu_blas_gemm.rs`

### Checkpoint 1.2: GEMM Shader (Hour 0.5-2)
- [ ] Implement basic GEMM in `wgpu_shaders/gemm.wgsl`
  - [ ] Use tile-based approach (16×16 workgroups)
  - [ ] Support M, N, K dimensions
  - [ ] Handle alpha/beta scaling
  - [ ] Support transpose flags (trans_a, trans_b)
  - [ ] Use shared memory for A and B tiles
  - [ ] Synchronize with `workgroupBarrier()`
- [ ] Validate syntax: `naga check src/domain/compute/wgpu_shaders/gemm.wgsl`
- [ ] No errors, only warnings acceptable

### Checkpoint 1.3: GEMM Rust Wrapper (Hour 2-3)
- [ ] Implement in `src/domain/compute/wgpu_ops.rs`:
  - [ ] Add `gemm_f32()` method to `WgpuMatrixOps`
  - [ ] Create shader module and pipeline
  - [ ] Create bind groups for A, B, C, params
  - [ ] Dispatch workgroups: `ceil(M/16) × ceil(N/16) × 1`
  - [ ] Handle parameter struct (GemmParams)
  - [ ] Upload params to uniform buffer
  - [ ] Synchronize queue after dispatch
- [ ] Compile test: `cargo test --lib --features wgpu --no-run`
- [ ] No compilation errors

### Checkpoint 1.4: GEMM Test Harness (Hour 3-4)
- [ ] Create tests in `tests/gpu_blas_gemm.rs`:
  - [ ] Test 1: Small identity matrix (4×4)
  - [ ] Test 2: Medium random (64×64)
  - [ ] Test 3: Large matrices (256×256)
  - [ ] Test 4: Alpha/beta scaling
  - [ ] Test 5: Transpose flags (NN, NT, TN, TT)
  - [ ] Test 6: CPU reference comparison (tolerance ε ≤ 1e-4)
  - [ ] Test 7-10: Edge cases (zeros, large values)
- [ ] Run tests: `cargo test --test gpu_blas_gemm --features wgpu`
- [ ] Expect: ≥50 test cases passing

### Checkpoint 1.5: Element-Wise Operations (Hour 4-5)
Implement in `wgpu_shaders/activation.wgsl` and Rust wrapper:
- [ ] `fill_f32` - Initialize buffer with scalar
- [ ] `scale` - Multiply by scalar (y *= scale)
- [ ] `mul` - Element-wise multiply (z = x * y)
- [ ] `relu` - Max(0, x)
- [ ] `sigmoid` - 1 / (1 + exp(-x)) (stable version)
- [ ] `gelu` - x * Φ(x) (approximation acceptable)

For each operation:
- [ ] Shader implemented
- [ ] Rust wrapper complete
- [ ] 3-5 test cases passing
- [ ] CPU reference validated

### Checkpoint 1.6: Normalization (Hour 5-6)
Implement in `wgpu_shaders/softmax.wgsl`:
- [ ] `layer_norm`:
  - [ ] Two-pass algorithm (mean, then variance)
  - [ ] Parallel reduction using shared memory
  - [ ] Apply gamma/beta scaling
  - [ ] 3+ test cases
- [ ] `softmax`:
  - [ ] Row-wise softmax
  - [ ] Log-sum-exp trick (numerically stable)
  - [ ] 3+ test cases

### Checkpoint 1.7: Validation (Hour 6-7)
- [ ] Run full test suite: `cargo test --lib --features wgpu`
- [ ] Expect: All 539 original + 100+ new GPU tests passing
- [ ] No compilation errors
- [ ] ≤3 warnings (deprecated CpuGpuMatrixOps expected)

### Checkpoint 1.8: Documentation & Commit (Hour 7-8)
- [ ] Add comments to WGSL shaders (algorithm description)
- [ ] Update shader implementations in code docs
- [ ] Test output recorded (test pass counts, timing)
- [ ] Benchmark run: `cargo bench --bench gpu_ops_benchmark --features wgpu`
- [ ] Git commit with template message

---

## Success Criteria Checklist

### Must Have (Required for Session 1 completion)
- [ ] GEMM shader compiles without errors
- [ ] GEMM test harness created (50+ tests)
- [ ] GEMM tests passing: 100% (all test cases pass)
- [ ] Element-wise ops: 6/6 implemented
- [ ] Element-wise tests: 18+ test cases passing
- [ ] Layer norm: implemented and tested
- [ ] Softmax: implemented and tested
- [ ] Total tests: 539 original + 100+ new = 640+ passing
- [ ] Compilation: `cargo test --lib --features wgpu` succeeds
- [ ] No clippy warnings in new GPU modules

### Nice to Have (Polish)
- [ ] Benchmarks recorded and analyzed
- [ ] Performance meets targets (GEMM >1 TFLOPS)
- [ ] Documentation complete (shader comments)
- [ ] Code formatted with `cargo fmt`
- [ ] Example test script for future validation

### Stretch Goals (If ahead of schedule)
- [ ] Batched GEMM started
- [ ] GEMV started
- [ ] PolyAttention kernels stub created
- [ ] Performance optimization (shared memory improvements)

---

## Hourly Breakdown (8-hour session)

| Hour | Task | Checkpoint |
|------|------|-----------|
| 0 | Review docs, setup | Project structure ready |
| 1-2 | GEMM shader | Shader compiles |
| 2-3 | GEMM wrapper | Rust code compiles |
| 3-4 | GEMM tests | 50+ tests passing |
| 4 | Element-wise ops | 6/6 implemented |
| 5 | Normalization | Layer norm + softmax done |
| 6 | Validation | 640+ tests passing |
| 7 | Final checks | Code clean, documented |
| 8 | Commit/handoff | Ready for Session 2 |

---

## Git Workflow

### At Session Start
```bash
git pull origin main
git checkout -b feature/gpu-phase5.5-blas
```

### Checkpoint Commits (Every 1-2 hours)
```bash
git add src/domain/compute/wgpu_shaders/ tests/gpu_blas_*.rs
git commit -m "WIP: GEMM shader implementation"

git add src/domain/compute/wgpu_ops.rs
git commit -m "WIP: GEMM Rust wrapper and test harness"

# ... more commits ...
```

### Session End
```bash
git add -A
git commit -m "feat(gpu): Phase 5.5 Session 1 - WGPU BLAS foundation complete"
git push origin feature/gpu-phase5.5-blas
# Then create PR or merge to main
```

---

## Key Files to Have Open

1. **WGPU_BLAS_IMPLEMENTATION_GUIDE.md** - Reference for algorithms
2. **src/domain/compute/gpu_ops.rs** - Trait definitions (GpuMatrixOps)
3. **src/domain/compute/wgpu_ops.rs** - Implementation file
4. **wgpu_shaders/gemm.wgsl** - Shader being edited
5. **tests/gpu_blas_gemm.rs** - Test harness
6. **tests/** (existing GPU tests) - Reference patterns

---

## Common Errors & Quick Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "Shader compilation failed" | WGSL syntax error | Run `naga check shader.wgsl` |
| "Pipeline creation failed" | Bind group layout mismatch | Check entry points and binding numbers |
| "Buffer overflow in shared memory" | Tile size too large | Reduce from 16×16 to 8×8 |
| "Test timeout" | Problem size too large | Use smaller matrices for debug builds |
| "GPU device not found" | GPU not available | Use CPU build or different machine |
| "Numerical accuracy low" | Tolerance too strict | Check tolerance formula: `abs(gpu - cpu) <= 1e-4 * abs(cpu)` |

---

## Resources at Your Fingertips

### WGSL Shader Language
- Full spec: https://www.w3.org/TR/WGSL/
- Quick reference: See WGPU_BLAS_IMPLEMENTATION_GUIDE.md section 2

### WGPU API
- Full docs: https://docs.rs/wgpu/
- Examples: https://github.com/gfx-rs/wgpu/tree/master/examples

### Project References
- CPU GEMM: `src/domain/compute/gpu_ops.rs` (search for gemm_f32)
- Existing GPU code: `src/domain/compute/wgpu_ops.rs`
- Test patterns: `tests/gpu_integration_*.rs`

---

## Post-Session Checklist

At end of session:
- [ ] All checkpoints completed
- [ ] 640+ tests passing
- [ ] Code committed to feature branch
- [ ] Benchmarks recorded
- [ ] Documentation updated
- [ ] Next session tasks identified
- [ ] Handoff notes written

---

## Session 2 Continuation

If making progress faster than expected:
- [ ] Start Task 1.4: PolyAttention kernels (stubs → implementation)
- [ ] Start Task 1.5: GEMV (matrix-vector multiply)
- [ ] Start Task 1.6: Batched GEMM (parallel independent multiplies)

Prioritize in order: GEMM → Element-wise → Normalization → Attention

---

**Session Start**: February 15, 2026, ~00:00 UTC  
**Expected End**: February 15, 2026, ~08:00 UTC (8 hours)  
**Status**: Fully prepared and documented

---

## Final Reminders

1. ✓ All prerequisite files created and tested
2. ✓ Documentation is thorough and actionable
3. ✓ Code infrastructure in place (wgpu_ops, gpu_ops, gpu_device)
4. ✓ Test patterns established from existing GPU tests
5. ✓ Environment can be prepared 30 min before session
6. ✓ Timeline is realistic (1 major operation per hour)
7. ✓ Rollback plan: If WGPU issues, switch to CPU testing
8. ✓ Stretch goals defined if ahead of schedule

**You are ready. Let's go! 🚀**
