# Quick Commands: Phase 5.6 GPU Consolidation

## Build & Verify

### Check Compilation (No GPU)
```bash
cd d:\RustGPT
cargo check --lib
```

### Build with GPU Support
```bash
# WGPU (portable)
cargo build --release --features wgpu

# CUDA (NVIDIA)
cargo build --release --features gpu-cuda

# Metal (macOS)
cargo build --release --features gpu-metal

# All backends
cargo build --release --features gpu-all
```

### Quick Verify After Changes
```bash
cargo fmt
cargo clippy --all-targets
cargo check --lib
```

---

## Testing

### Run All Tests
```bash
cargo test --lib
```

### Run GPU Tests (WGPU)
```bash
cargo test --lib --features wgpu
```

### Run Specific Test
```bash
cargo test --lib test_gpu_forward_dispatch -- --exact --nocapture
```

### Run Tests with Output
```bash
cargo test --lib -- --nocapture --test-threads=1
```

---

## Documentation Workflow

### Locate Key Documents
```bash
# Priority 1-2 Summary
cat README_PHASE5_6_GPU_CONSOLIDATION_FEB15.md

# Priority 3.1 Status
cat PHASE5_6_PRIORITY3_GPU_INFRASTRUCTURE_COMPLETE.md

# Quick Reference
cat QUICK_REFERENCE_SHAREDFEEDFORWARD_GPU_WIRING.md

# Implementation Plan
cat PHASE5_6_PRIORITY3_POLYATTENTION_GPU_PLAN.md
```

### Quick Status
```bash
ls -la PHASE5_6_*.md SESSION_FEB15_*.md GPU_KERNEL_*.md
```

---

## Code Navigation

### Find GPU-Related Code
```bash
# Richards GPU kernels
grep -r "apply_richards_activation_gpu" src/

# PolyAttention GPU implementation
grep -r "GpuComponent for PolyAttention" src/

# GPU device methods
grep -n "pub fn.*gpu" src/domain/compute/gpu_device.rs | head -20

# GPU trait definitions
grep -n "trait GpuMatrixOps" src/domain/compute/gpu_ops.rs
```

### View Implementation
```bash
# Richards GPU kernel dispatch
code src/domain/compute/richards_glu_fused_kernel.rs:215-269

# PolyAttention GpuComponent impl
code src/domain/attention/poly_attention.rs:3267-3339

# GpuDevice API
code src/domain/compute/gpu_device.rs:196-201
```

---

## Development Workflow

### Start New Phase (e.g., Phase 3.2)
```bash
# 1. Read the plan
cat PHASE5_6_PRIORITY3_POLYATTENTION_GPU_PLAN.md

# 2. Create a new implementation branch (if using git)
git checkout -b phase-3.2-poly-gpu

# 3. Open the file to edit
code src/domain/attention/poly_attention.rs

# 4. Implement changes following the plan

# 5. Verify compilation
cargo check --lib

# 6. Test
cargo test --lib --features wgpu

# 7. Document
echo "## Phase 3.2 Completion" > PHASE5_6_PRIORITY3_GPU_KERNELS_COMPLETE.md
```

### Integrate GPU Kernels
```bash
# 1. Locate the component
grep -n "pub struct PolyAttention {" src/domain/attention/poly_attention.rs

# 2. Find forward method
grep -n "pub fn forward" src/domain/attention/poly_attention.rs

# 3. Add forward_gpu() method
# (after existing forward methods)

# 4. Test compilation
cargo check --lib --features wgpu

# 5. Run unit tests
cargo test --lib test_poly_attention --features wgpu
```

---

## File Structure Reference

### Core GPU Implementation
```
src/domain/compute/
├── gpu_device.rs              # GPU device abstraction
├── gpu_ops.rs                 # GPU kernel trait definitions
├── gpu_component.rs           # Unified component trait
├── gpu_memory.rs              # Memory pool management
├── wgpu_ops.rs               # WGPU backend
├── cuda/                      # CUDA backend
└── metal/                     # Metal backend

src/domain/
├── richards/
│   ├── richards_glu.rs        # RichardsGlu with GPU support
│   └── richards_curve.rs      # Richards activation formula
├── attention/
│   ├── poly_attention.rs      # PolyAttention with GPU infrastructure (Phase 3.1)
│   └── poly_attention_gpu.rs  # GPU helpers (Phase 3.2+)
└── layers/components/
    ├── feedforward.rs         # SharedFeedforward (GPU integrated)
    └── temporal_processing.rs # (Future GPU support)
```

### Documentation
```
Documentation/
├── GPU_KERNEL_WIRING_PHASE5_6.md                    # Technical reference
├── QUICK_REFERENCE_SHAREDFEEDFORWARD_GPU_WIRING.md  # Quick lookup
├── PHASE5_6_GPU_KERNEL_CONSOLIDATION_SUMMARY.md    # Summary
├── PHASE5_6_PRIORITY3_POLYATTENTION_GPU_PLAN.md    # Roadmap
├── PHASE5_6_PRIORITY3_GPU_INFRASTRUCTURE_COMPLETE.md # Phase 3.1
├── SESSION_FEB15_GPU_KERNEL_WIRING_START.md        # Session start
├── SESSION_FEB15_FINAL_STATUS.md                    # Session end
├── README_PHASE5_6_GPU_CONSOLIDATION_FEB15.md      # Main summary
├── INDEX_PHASE5_6_GPU_CONSOLIDATION_FEB15.md       # Index
└── QUICK_COMMANDS_PHASE5_6.md                       # This file
```

---

## Common Tasks

### Add GPU Method to Component
```rust
// 1. Add to struct
pub struct MyComponent {
    #[serde(skip)]
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,
    // ... other fields
}

// 2. Initialize in new()
Self {
    gpu_device: None,
    // ... other fields
}

// 3. Implement GpuComponent trait
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuComponent for MyComponent {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.gpu_device = Some(device);
    }
    // ... other trait methods
}

// 4. Implement forward_gpu()
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let device_arc = self.gpu_device.as_ref()
        .ok_or(ModelError::Backend { ... })?
        .clone();
    
    let mut device = device_arc.lock().unwrap();
    
    // GPU computation here
    
    Ok(output)
}
```

### Test GPU Path
```rust
#[test]
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn test_gpu_forward() {
    if let Ok(mut device) = GpuDevice::auto_detect() {
        let mut component = MyComponent::new(...);
        component.set_gpu_device(Arc::new(Mutex::new(device)));
        
        let input = Array2::random((batch, dim));
        let output = component.forward_gpu(&input).unwrap();
        
        assert_eq!(output.dim(), (batch, dim));
    }
}
```

---

## Debug Commands

### Check GPU Device Detection
```bash
# In code:
match GpuDevice::auto_detect() {
    Ok(device) => println!("GPU available: {}", device.name()),
    Err(e) => println!("GPU error: {}", e),
}
```

### Verify Parameter Types
```bash
# Find parameter types
grep -n "RichardsCurveParams\|OptimizedRichardsGluParams" src/domain/compute/*.rs
```

### Check Memory Leaks
```bash
# Run with leak detection (on Unix)
RUSTFLAGS="-Z sanitizer=leak" cargo +nightly test --lib
```

### Profile GPU Operations
```bash
# Release build with symbols
cargo build --release --lib

# Time specific test
time cargo test --lib test_gpu_forward_dispatch --release
```

---

## Compilation Issues

### Common Errors & Solutions

**Error**: `feature "wgpu" not found`
```bash
# Solution: Enable feature in Cargo.toml or build command
cargo build --release --features wgpu
```

**Error**: `GpuComponent not in scope`
```rust
// Solution: Add import
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::GpuComponent;
```

**Error**: `Arc<Mutex<GpuDevice>> not found`
```rust
// Solution: Add imports
use std::sync::{Arc, Mutex};
use crate::domain::compute::GpuDevice;
```

**Error**: `unused import` warning
```rust
// Solution: Conditional import
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::Result;
```

---

## Reference Links

### Documentation Quick Access
```bash
# Main entry point
cat README_PHASE5_6_GPU_CONSOLIDATION_FEB15.md

# Quick lookup (5 min)
cat QUICK_REFERENCE_SHAREDFEEDFORWARD_GPU_WIRING.md

# Full technical reference (30+ min)
cat GPU_KERNEL_WIRING_PHASE5_6.md

# Phase 3.1 Details
cat PHASE5_6_PRIORITY3_GPU_INFRASTRUCTURE_COMPLETE.md

# Next steps planning
cat PHASE5_6_PRIORITY3_POLYATTENTION_GPU_PLAN.md
```

---

## Environment Setup

### Windows 10 (Current Session Environment)
```bash
cd d:\RustGPT
rustc --version  # Should be 1.85+
cargo --version  # Should be recent
```

### Build Cache Management
```bash
# Clean build
cargo clean

# Clean specific target
cargo clean --target-dir target

# Faster incremental builds
cargo build

# Full release build
cargo build --release
```

---

## Git Workflow (if using version control)

```bash
# Create branch for phase
git checkout -b phase-3.2-poly-attention-gpu

# After changes, check diff
git diff src/domain/attention/poly_attention.rs

# Commit phase completion
git commit -am "Phase 3.2: PolyAttention GPU kernels"

# Push to remote
git push origin phase-3.2-poly-attention-gpu
```

---

## Performance Profiling

### Quick Benchmark
```bash
# Build release binary
cargo build --release --features wgpu

# Time a test
/usr/bin/time cargo test --lib test_gpu_forward_dispatch --release
```

### Compare CPU vs GPU
```bash
# CPU version
time cargo test --lib test_forward_dispatch

# GPU version
time cargo test --lib test_gpu_forward_dispatch --features wgpu
```

---

## Session Checklist

### Before Ending Session
- [ ] Code compiles: `cargo check --lib`
- [ ] Tests pass: `cargo test --lib`
- [ ] GPU tests pass: `cargo test --lib --features wgpu`
- [ ] Documentation updated
- [ ] Next phase documented in a plan file
- [ ] Handoff notes created

### For Next Session
- [ ] Read the plan file (e.g., `PHASE5_6_PRIORITY3_POLYATTENTION_GPU_PLAN.md`)
- [ ] Check if code compiles: `cargo check --lib`
- [ ] Review previous phase completion document
- [ ] Start implementing Phase 3.2

---

## Summary

**Phase 5.6 GPU Consolidation** - Command Quick Reference

Use these commands to navigate, build, test, and develop GPU-accelerated components.

For detailed technical information, refer to the comprehensive documentation files listed above.

---

**Status**: Ready for Phase 3.2+ Development  
**Next**: PolyAttention GPU Kernel Implementation  
**Estimated Time**: 4-6 hours for Phase 3.2-3.5
