# GPU Consolidation Migration Guide (Phase 5.4)

**For**: Developers migrating from legacy GPU managers  
**Status**: Active (Phase 5.4)  
**Deprecation Timeline**: Phase 5.4 (current) → Phase 6 (removal)

---

## Quick Start: Implementing GPU Support for a Component

### Old Way (Deprecated - Phase 5.3)
```rust
// Don't do this anymore - use consolidated API
use crate::domain::layers::components::shared_gpu_manager::{SharedComponentGpuManager, GpuComponent};

pub struct MyComponent {
    gpu_manager: SharedComponentGpuManager,  // ❌ Deprecated
    // ...
}

impl MyComponent {
    fn new() -> Self {
        Self {
            gpu_manager: SharedComponentGpuManager::new(),
            // ...
        }
    }
    
    pub fn enable_gpu(&mut self) -> Result<()> {
        self.gpu_manager.enable_gpu_auto_detect()
    }
}
```

### New Way (Phase 5.4+)
```rust
// ✅ Use unified GPU API from compute module
use crate::domain::compute::{GpuComponent, GpuDevice};
use std::sync::{Arc, Mutex};

pub struct MyComponent {
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,  // ✅ Unified
    batch_size: usize,
    embed_dim: usize,
    seq_len: usize,
}

impl MyComponent {
    fn new() -> Self {
        Self {
            gpu_device: None,
            batch_size: 0,
            embed_dim: 0,
            seq_len: 0,
        }
    }
}

impl GpuComponent for MyComponent {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.gpu_device = Some(device);
    }
    
    fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;  // Strict: errors if no GPU
        self.gpu_device = Some(Arc::new(Mutex::new(device)));
        Ok(())
    }
    
    fn is_gpu_ready(&self) -> bool {
        self.gpu_device.is_some()
    }
    
    fn gpu_backend_name(&self) -> Option<&'static str> {
        self.gpu_device.as_ref()
            .and_then(|d| d.lock().ok())
            .map(|guard| guard.backend().as_str())
    }
    
    fn ensure_capacity(&mut self, batch_size: usize, embed_dim: usize, seq_len: usize) -> Result<()> {
        self.batch_size = batch_size;
        self.embed_dim = embed_dim;
        self.seq_len = seq_len;
        Ok(())
    }
}
```

---

## Migration Checklist

For each component that uses the old GPU managers:

### 1. Remove Old Imports
```rust
// ❌ Remove these
use crate::domain::layers::components::shared_gpu_manager::{SharedComponentGpuManager, GpuComponent};
use crate::domain::layers::components::gpu_shared_ops::{GpuSharedOpsContext, GpuBatchExecutor};

// ✅ Add these
use crate::domain::compute::{GpuComponent, GpuDevice};
use std::sync::{Arc, Mutex};
```

### 2. Update Struct Fields
```rust
// ❌ Old
pub struct MyBlock {
    gpu_manager: SharedComponentGpuManager,
    gpu_context: GpuSharedOpsContext,
}

// ✅ New
pub struct MyBlock {
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,
    // ... other fields
}
```

### 3. Implement GpuComponent Trait
```rust
impl GpuComponent for MyBlock {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.gpu_device = Some(device);
    }
    
    fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;
        self.gpu_device = Some(Arc::new(Mutex::new(device)));
        Ok(())
    }
    
    fn is_gpu_ready(&self) -> bool {
        self.gpu_device.is_some()
    }
    
    fn gpu_backend_name(&self) -> Option<&'static str> {
        self.gpu_device.as_ref()
            .and_then(|d| d.lock().ok())
            .map(|guard| guard.backend().as_str())
    }
    
    fn ensure_capacity(&mut self, batch_size: usize, embed_dim: usize, seq_len: usize) -> Result<()> {
        // Validate or prepare buffers if needed
        Ok(())
    }
}
```

### 4. Update GPU Forward Implementation
```rust
// ❌ Old pattern
pub fn forward_gpu(&mut self, input: &Array2<f32>, ctx: &mut GpuSharedOpsContext) -> Result<Array2<f32>> {
    ctx.ensure_capacity(batch_size, embed_dim, seq_len)?;
    // ... GPU computation
}

// ✅ New pattern
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // Validate GPU ready
    require_gpu_device(&self.gpu_device, "forward_gpu")?;
    
    let (batch_size, embed_dim) = (input.nrows(), input.ncols());
    self.ensure_capacity(batch_size, embed_dim, 1)?;
    
    // ... GPU computation using GpuDevice API
}
```

### 5. Update Tests
```rust
// ❌ Old
#[test]
fn test_gpu_forward() {
    let mut block = MyBlock::new();
    let mut ctx = GpuSharedOpsContext::new();
    ctx.enable_gpu_auto_detect().ok();
    
    let input = Array2::random((2, 64));
    let output = block.forward_gpu(&input, &mut ctx);
}

// ✅ New
#[test]
fn test_gpu_forward() {
    let mut block = MyBlock::new();
    if let Err(e) = block.enable_gpu_auto_detect() {
        println!("No GPU available: {}", e);
        return;  // Skip if no GPU
    }
    
    let input = Array2::random((2, 64));
    let output = block.forward_gpu(&input);
}
```

---

## API Reference

### GpuComponent Trait

```rust
pub trait GpuComponent: Sized {
    /// Attach a pre-created GPU device
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>);
    
    /// Auto-detect and attach GPU (strict: errors if no GPU available)
    fn enable_gpu_auto_detect(&mut self) -> Result<()>;
    
    /// Check if GPU is ready for computation
    fn is_gpu_ready(&self) -> bool;
    
    /// Get GPU backend name ("CUDA", "Metal", "Vulkan", etc.)
    fn gpu_backend_name(&self) -> Option<&'static str>;
    
    /// Ensure buffers are sized for given dimensions
    fn ensure_capacity(&mut self, batch_size: usize, embed_dim: usize, seq_len: usize) -> Result<()>;
}
```

### Helper Functions

#### require_gpu_device()
```rust
use crate::domain::compute::require_gpu_device;

fn validate_gpu_ready(device: &Option<Arc<Mutex<GpuDevice>>>) -> Result<Arc<Mutex<GpuDevice>>> {
    require_gpu_device(device, "my_operation")
    // Returns: Arc<Mutex<GpuDevice>> or error with clear message
}
```

### GpuDevice API

```rust
use crate::domain::compute::GpuDevice;

// Auto-detect GPU (strict: errors if none available)
let device = GpuDevice::auto_detect()?;

// Or create for specific backend
let device = GpuDevice::new(ComputeBackend::Vulkan)?;

// Access to GPU operations
let (memory, ops) = device.execution_context();
ops.gemm_f32(1.0, &a, &b, 0.0, &mut c, m, n, k)?;

// Memory management
device.allocate_f32(1024)?;
device.upload(&cpu_data, &mut gpu_buffer)?;
device.download(&gpu_buffer, &mut cpu_data)?;
```

---

## Strict No-Fallback Behavior

The consolidated GPU API enforces strict no-fallback mode:

### What Changed
```rust
// ❌ Old: Silent fallback to CPU
pub fn forward_gpu(&mut self, input) {
    if !self.is_gpu_ready {
        return self.forward_cpu(input);  // Silent fallback!
    }
    // ... GPU code
}

// ✅ New: Explicit error
pub fn forward_gpu(&mut self, input) -> Result<Array2<f32>> {
    require_gpu_device(&self.gpu_device, "forward_gpu")?;  // Error if no GPU
    // ... GPU code
}
```

### Caller Responsibility
The **caller** decides what to do if GPU is unavailable:

```rust
// Caller chooses behavior
let output = match block.forward_gpu(&input) {
    Ok(result) => result,
    Err(e) if is_gpu_error(&e) => {
        println!("GPU unavailable, falling back to CPU");
        block.forward_cpu(&input)?  // Explicit fallback
    }
    Err(e) => return Err(e),
};
```

### Benefits
- **Predictable**: No silent performance degradation
- **Debuggable**: Clear error messages if GPU unavailable
- **Flexible**: Components control fallback strategy
- **Observable**: Can track GPU usage patterns

---

## Common Migration Patterns

### Pattern 1: Component with GPU Support
```rust
pub struct MyComponent {
    // Device management
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,
    
    // Capacity tracking
    batch_size: usize,
    embed_dim: usize,
    
    // Other fields
    weights: Array2<f32>,
}

impl MyComponent {
    pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if self.is_gpu_ready() {
            self.forward_gpu(input)
        } else {
            self.forward_cpu(input)
        }
    }
    
    pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        require_gpu_device(&self.gpu_device, "forward_gpu")?;
        // ... GPU implementation
    }
    
    pub fn forward_cpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // ... CPU implementation
    }
}

impl GpuComponent for MyComponent {
    // Implement trait...
}
```

### Pattern 2: Pipeline with GPU
```rust
pub struct Pipeline {
    blocks: Vec<Box<dyn GpuComponent>>,
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,
}

impl Pipeline {
    pub fn enable_gpu_all(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;
        let device_arc = Arc::new(Mutex::new(device));
        
        for block in &mut self.blocks {
            block.set_gpu_device(device_arc.clone());
        }
        
        Ok(())
    }
    
    pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let mut output = input.to_owned();
        for block in &mut self.blocks {
            output = if block.is_gpu_ready() {
                block.forward_gpu(&output)?
            } else {
                block.forward_cpu(&output)?
            };
        }
        Ok(output)
    }
}
```

---

## Troubleshooting

### Issue: "GPU operation requested without GPU device attached"
**Cause**: Calling `forward_gpu()` before `enable_gpu_auto_detect()`  
**Fix**: Call `enable_gpu_auto_detect()` first or check `is_gpu_ready()`
```rust
block.enable_gpu_auto_detect()?;
let output = block.forward_gpu(&input)?;
```

### Issue: "Automatic GPU detection failed: no supported GPU backend was detected"
**Cause**: No GPU available on system or no matching feature flags  
**Fix**: Either use CPU mode or compile with GPU features
```bash
# Enable GPU support
cargo build --release --features gpu-wgpu

# Or fall back to CPU
let output = block.forward_cpu(&input)?;
```

### Issue: GPU computation returns different results than CPU
**Cause**: Numerical precision differences or unimplemented GPU kernel  
**Fix**: Check GPU kernel implementation and numerical tolerance
```rust
let max_diff = (gpu_output - cpu_output).mapv(f32::abs).max();
assert!(max_diff < 1e-4, "GPU output differs from CPU");
```

---

## Complete Example: TransformerBlock Migration

See: `PHASE5.4_GPU_FORWARD_IMPLEMENTATION_GUIDE.md` - Section 3 for detailed TransformerBlock example

---

## FAQ

**Q: Do I have to migrate immediately?**  
A: No, old managers are still supported through Phase 5. Migrate during Phase 5.4 to avoid code churn in Phase 6.

**Q: What if my component doesn't support GPU yet?**  
A: Implement `GpuComponent` with `is_gpu_ready()` returning `false` and add GPU support later.

**Q: Can I share a GPU device across components?**  
A: Yes! Use `Arc<Mutex<GpuDevice>>` and call `set_gpu_device()` on each component.

**Q: How do I benchmark GPU vs CPU?**  
A: Use the `forward_gpu()` and `forward_cpu()` methods with standard Rust benchmarks.

---

## References

- Consolidation Plan: SESSION_CONSOLIDATION_GPU_PHASE5.4_PLAN.md
- Implementation Guide: PHASE5.4_GPU_FORWARD_IMPLEMENTATION_GUIDE.md
- GPU Architecture: GPU_BACKEND_IMPLEMENTATION_STATUS.md
