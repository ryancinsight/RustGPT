# Quick Start: Phase 5.6.1b - GPU Integration in Components

**Status**: Cleanup complete ✅, Ready for component integration 🚀

---

## What to Do Right Now

### Step 1: Verify Baseline (5 min)
```bash
# Check compilation - should be instant, 0 warnings
cargo check --lib

# Run tests - should pass all 548
cargo test --lib

# Verify GPU features work
cargo check --lib --features gpu-wgpu
```

**Expected Output**: ✅ All pass, 0 warnings

---

### Step 2: Wire SharedAttentionContext (30 min)

**File**: `src/domain/layers/components/attention_context.rs`

**Add to struct**:
```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
gpu_backend: Option<Arc<Mutex<UnifiedGpuBackend>>>,
```

**Add method**:
```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn enable_gpu_auto_detect(&mut self) -> Result<()> {
    let backend = UnifiedGpuBackend::auto_detect()?;
    self.gpu_backend = Some(Arc::new(Mutex::new(backend)));
    Ok(())
}
```

**Update apply_incoming_context()**:
```rust
pub fn apply_incoming_context(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    if let Some(backend) = &self.gpu_backend {
        let mut guard = backend.lock().map_err(|_| ModelError::Backend {
            message: "GPU backend lock failed".to_string(),
        })?;
        return self.apply_incoming_context_gpu(input, &mut guard);
    }
    self.apply_incoming_context_cpu(input)
}
```

**Test**:
```bash
cargo check --lib
cargo test --lib attention_context
```

---

### Step 3: Wire SharedFeedforward (30 min)

**File**: `src/domain/layers/components/shared_feedforward.rs`

Same pattern as SharedAttentionContext:
1. Add `gpu_backend: Option<...>` field
2. Add `enable_gpu_auto_detect()` method
3. Update `forward()` to check GPU first

---

### Step 4: Wire SharedTemporalProcessing (30 min)

**File**: `src/domain/layers/components/shared_temporal_processing.rs`

Same pattern, but note: Dispatch to different GPU kernels based on temporal type (Attention, Mamba, RgLru)

---

### Step 5: Create Integration Tests (30 min)

**File**: `tests/gpu_integration_basic.rs`

```rust
#[cfg(all(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"), test))]
mod gpu_integration_tests {
    use llm::domain::layers::components::unified_gpu_backend::UnifiedGpuBackend;
    use llm::domain::layers::components::attention_context::SharedAttentionContext;
    use ndarray::Array2;

    #[test]
    fn test_attention_context_gpu_enable() {
        let mut context = SharedAttentionContext::new();
        match context.enable_gpu_auto_detect() {
            Ok(()) => println!("✓ GPU enabled"),
            Err(e) => println!("✓ GPU unavailable: {}", e),
        }
    }

    #[test]
    fn test_gpu_auto_detect_strict() {
        match UnifiedGpuBackend::auto_detect() {
            Ok(backend) => {
                println!("✓ GPU: {}", backend.backend_name());
                assert!(backend.is_ready());
            }
            Err(e) => {
                println!("✓ No GPU (expected): {}", e);
            }
        }
    }
}
```

---

## Key Patterns

### Pattern 1: GPU Field + Enable Method
```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
gpu_backend: Option<Arc<Mutex<UnifiedGpuBackend>>>,

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn enable_gpu_auto_detect(&mut self) -> Result<()> {
    let backend = UnifiedGpuBackend::auto_detect()?;
    self.gpu_backend = Some(Arc::new(Mutex::new(backend)));
    Ok(())
}
```

### Pattern 2: GPU-First Forward
```rust
pub fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    if let Some(backend) = &self.gpu_backend {
        let mut guard = backend.lock().map_err(|_| ModelError::Backend {
            message: "Lock failed".to_string(),
        })?;
        return self.forward_gpu(input, &mut guard);
    }
    
    self.forward_cpu(input)
}
```

### Pattern 3: Conditional Import
```rust
use crate::domain::layers::components::unified_gpu_backend::UnifiedGpuBackend;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::unified_gpu_backend::GpuActivation;
```

---

## Testing Checklist

- [ ] `cargo check --lib` - 0 warnings
- [ ] `cargo test --lib` - All pass
- [ ] `cargo check --lib --features gpu-wgpu` - Compiles
- [ ] `cargo test --lib --features gpu-wgpu` - GPU tests run
- [ ] SharedAttentionContext can enable GPU
- [ ] SharedFeedforward can enable GPU
- [ ] SharedTemporalProcessing can enable GPU
- [ ] Clear error if GPU unavailable
- [ ] No silent CPU fallback

---

## Build Commands

```bash
# Check for warnings (should be 0)
cargo check --lib

# Full build
cargo build --lib

# With WGPU GPU support
cargo check --lib --features gpu-wgpu
cargo build --lib --features gpu-wgpu

# Run tests
cargo test --lib
cargo test --lib --features gpu-wgpu

# Run specific test
cargo test --lib attention_context_gpu -- --exact

# Format code
cargo fmt

# Check code style
cargo clippy --all-targets
```

---

## Expected Outcomes

By end of this session:

✅ SharedAttentionContext supports GPU auto-detect  
✅ SharedFeedforward supports GPU auto-detect  
✅ SharedTemporalProcessing supports GPU auto-detect  
✅ All tests pass (CPU and GPU)  
✅ Zero compiler warnings  
✅ Clear error messages when GPU unavailable  
✅ No silent CPU fallback  

---

## Files to Modify

1. `src/domain/layers/components/attention_context.rs` - Add GPU backend
2. `src/domain/layers/components/shared_feedforward.rs` - Add GPU backend
3. `src/domain/layers/components/shared_temporal_processing.rs` - Add GPU backend
4. `tests/gpu_integration_basic.rs` - Create new GPU integration tests

---

## If You Get Stuck

1. **Check GPU is available**: `nvidia-smi` or `metal-gpu-metrics`
2. **Verify feature flags**: `cargo check --lib --features gpu-wgpu`
3. **Check imports**: Ensure `UnifiedGpuBackend` is imported under feature gate
4. **Check Arc<Mutex>**: Use `.lock().map_err(...)` for error handling
5. **Check compilation**: `cargo clippy --all-targets` for suggestions

---

## Time Estimate

- SharedAttentionContext: 30 min
- SharedFeedforward: 30 min
- SharedTemporalProcessing: 30 min
- Tests: 30 min
- **Total**: 2 hours

**Total with buffer**: 2.5-3 hours

---

## Success = 

✅ All components can enable GPU  
✅ GPU auto-detect works (or clear error)  
✅ Tests pass (CPU and GPU)  
✅ Zero warnings  
✅ Code committed to Phase 5.6.1b branch  

**Ready for Phase 5.6.2** (Kernel Implementation + Buffer Pool)

