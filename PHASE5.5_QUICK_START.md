# Phase 5.5 GPU Consolidation - Quick Start Guide

**Goal**: Get started with UnifiedGpuBufferPool and GpuComponent trait implementation

**Time to First Code**: 5 minutes  
**Time to First Test**: 30 minutes

---

## Step 1: Understand the Architecture (5 min)

**Key Concepts**:
- **UnifiedGpuBufferPool**: Centralized GPU device + buffer management
- **GpuComponent trait**: Standard interface for GPU-capable components
- **Strict no-fallback**: All GPU operations error explicitly, no silent fallbacks
- **Auto-detection**: `auto_detect()` returns `Result`, fails if no GPU available

**Current State**:
```
Before (Fragmented)
├── SharedComponentGpuManager (in components/)
├── GpuSharedOpsContext (in components/)
├── Individual component GPU managers
└── Duplicate buffer allocation logic

After (Consolidated) → UnifiedGpuBufferPool
├── Single device management
├── Centralized buffer caching
├── Standard GpuComponent trait
└── 0 duplicate code
```

---

## Step 2: Set Up Environment (5 min)

### Check Current Status
```bash
cd d:\RustGPT

# Quick check (should complete in <30s if no changes)
cargo check --lib 2>&1 | head -20

# If slow, kill with Ctrl+C and try:
# (The build is genuinely slow due to project size)
```

### Optional: Speed Up Future Builds
```bash
# Use incremental compilation
export CARGO_INCREMENTAL=1

# Or use just for fast checks
cargo install just

# Then: just check
```

---

## Step 3: Create UnifiedGpuBufferPool (START HERE)

**File to Create**: `src/domain/compute/unified_gpu_buffer_pool.rs`

**Skeleton**:
```rust
//! Unified GPU Buffer Pool
//!
//! Centralized GPU device and buffer management for all components.
//! Replaces: SharedComponentGpuManager, GpuSharedOpsContext
//!
//! # Strict No-Fallback Design
//! 
//! All operations that require GPU return Result and error clearly if GPU unavailable.
//! There is no silent fallback to CPU.

use crate::common::errors::{ModelError, Result};
use super::gpu_memory::{GpuBuffer, GpuMemoryPool};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Unique identifier for memory pool instances
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct PoolId(usize);

/// Buffer specification for allocation requests
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct BufferSpec {
    /// Requested size in bytes
    pub size: usize,
    /// Required alignment (power-of-2 preferred)
    pub alignment: usize,
}

impl BufferSpec {
    /// Create buffer spec with power-of-2 alignment
    pub fn new(size: usize) -> Self {
        let alignment = size.next_power_of_two();
        Self { size, alignment }
    }
    
    /// Actual allocated size with padding
    pub fn padded_size(&self) -> usize {
        ((self.size + self.alignment - 1) / self.alignment) * self.alignment
    }
}

/// GPU capacity tracking for batch processing
#[derive(Debug, Clone, Copy)]
pub struct GpuCapacity {
    pub max_batch_size: usize,
    pub max_seq_length: usize,
    pub max_embedding_dim: usize,
}

impl Default for GpuCapacity {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_seq_length: 2048,
            max_embedding_dim: 2048,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub total_available: usize,
    pub utilization_percent: f32,
    pub cached_buffers: usize,
}

/// Unified GPU buffer pool with automatic device detection
pub struct UnifiedGpuBufferPool {
    /// GPU device (Arc<Mutex> for thread-safe sharing)
    device: Arc<Mutex<GpuDevice>>,
    /// Memory pools managed by different backends
    memory_pools: HashMap<PoolId, Arc<Mutex<dyn GpuMemoryPool>>>,
    /// LRU cache of allocated buffers
    buffer_cache: HashMap<BufferSpec, Vec<Arc<GpuBuffer>>>,
    /// Current capacity tracking
    capacity: GpuCapacity,
    /// Statistics
    stats: MemoryStats,
}

impl UnifiedGpuBufferPool {
    /// Create pool with explicit device
    pub fn with_device(device: GpuDevice) -> Self {
        Self {
            device: Arc::new(Mutex::new(device)),
            memory_pools: HashMap::new(),
            buffer_cache: HashMap::new(),
            capacity: GpuCapacity::default(),
            stats: MemoryStats {
                total_allocated: 0,
                total_available: 0,
                utilization_percent: 0.0,
                cached_buffers: 0,
            },
        }
    }
    
    /// Automatic GPU detection with strict no-fallback
    ///
    /// # Errors
    /// 
    /// Returns error if:
    /// - No GPU device is available
    /// - CUDA feature disabled and no alternative backend
    /// - GPU driver initialization failed
    pub fn auto_detect() -> Result<Self> {
        // Try to detect GPU device
        let device = GpuDevice::auto_detect()
            .map_err(|_| ModelError::Backend {
                message: "Automatic GPU detection failed: no supported GPU backend detected. \
                          Try specifying --features gpu-cuda or gpu-wgpu during build.".to_string(),
            })?;
        
        Ok(Self::with_device(device))
    }
    
    /// Allocate buffer with automatic power-of-2 sizing
    pub fn allocate(&mut self, spec: BufferSpec) -> Result<Arc<GpuBuffer>> {
        // Try cache first
        if let Some(cached) = self.buffer_cache.get_mut(&spec) {
            if !cached.is_empty() {
                return Ok(cached.remove(0));
            }
        }
        
        // Allocate from device memory
        let device = self.device.lock().unwrap();
        let padded_spec = BufferSpec {
            size: spec.padded_size(),
            alignment: spec.alignment,
        };
        
        // This would call device.allocate(padded_spec)
        // For now: stub that returns error
        Err(ModelError::Backend {
            message: "GPU allocation not yet implemented - update with device.allocate()".to_string(),
        })
    }
    
    /// Get or allocate buffer
    pub fn get_or_allocate(&mut self, spec: BufferSpec) -> Result<Arc<GpuBuffer>> {
        self.allocate(spec)
    }
    
    /// Update capacity for current batch
    pub fn update_capacity(&mut self, batch_size: usize, seq_len: usize, embed_dim: usize) {
        self.capacity = GpuCapacity {
            max_batch_size: batch_size,
            max_seq_length: seq_len,
            max_embedding_dim: embed_dim,
        };
    }
    
    /// Get memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        self.stats
    }
    
    /// Get device reference
    pub fn device(&self) -> Arc<Mutex<GpuDevice>> {
        Arc::clone(&self.device)
    }
}

// Placeholder for GpuDevice - would come from compute::gpu_device
pub struct GpuDevice;

impl GpuDevice {
    pub fn auto_detect() -> Result<Self> {
        Err(ModelError::Backend {
            message: "GpuDevice::auto_detect not yet linked".to_string(),
        })
    }
}
```

**Next**: Link this to actual `GpuDevice` from `compute::gpu_device` module

---

## Step 4: Create GpuComponent Trait (10 min)

**File to Create**: `src/domain/compute/gpu_component.rs`

**Skeleton**:
```rust
//! GPU Component Trait
//!
//! Standard interface for components that can leverage GPU acceleration.
//! Implement this trait to enable automatic GPU device attachment and detection.

use crate::common::errors::Result;
use super::unified_gpu_buffer_pool::UnifiedGpuBufferPool;
use std::sync::{Arc, Mutex};

/// Standard interface for GPU-capable components
pub trait GpuComponent: Send + Sync {
    /// Attach external GPU device from pool
    fn attach_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) -> Result<()>;
    
    /// Detach GPU device (reverts to CPU)
    fn detach_gpu_device(&mut self);
    
    /// Check if GPU is ready and available
    fn gpu_ready(&self) -> Result<()>;
    
    /// Enable automatic GPU detection (strict: errors if no GPU)
    fn enable_gpu_auto_detect(&mut self) -> Result<()>;
}

// Placeholder
pub struct GpuDevice;
```

---

## Step 5: Link in compute/mod.rs (5 min)

**File**: `src/domain/compute/mod.rs`

**Add**:
```rust
pub mod unified_gpu_buffer_pool;
pub mod gpu_component;

pub use unified_gpu_buffer_pool::{UnifiedGpuBufferPool, BufferSpec, GpuCapacity};
pub use gpu_component::GpuComponent;
```

---

## Step 6: Run First Test (10 min)

**Create**: `tests/unified_gpu_buffer_pool_basics.rs`

```rust
//! Basic tests for UnifiedGpuBufferPool

#[cfg(test)]
mod tests {
    use rustgpt::domain::compute::UnifiedGpuBufferPool;

    #[test]
    fn test_buffer_spec_padding() {
        let spec = rustgpt::domain::compute::BufferSpec::new(100);
        assert!(spec.padded_size() >= spec.size);
        assert_eq!(spec.padded_size() % spec.alignment, 0);
    }

    #[test]
    fn test_gpu_capacity_defaults() {
        let cap = rustgpt::domain::compute::GpuCapacity::default();
        assert!(cap.max_batch_size > 0);
        assert!(cap.max_seq_length > 0);
        assert!(cap.max_embedding_dim > 0);
    }

    #[test]
    fn test_auto_detect_strict_no_fallback() {
        // Should error if no GPU (no silent fallback)
        match UnifiedGpuBufferPool::auto_detect() {
            Ok(_) => {
                // GPU available, that's fine
                println!("GPU detected successfully");
            }
            Err(e) => {
                // Should get clear error message
                let msg = format!("{:?}", e);
                assert!(msg.contains("failed") || msg.contains("not available"),
                    "Error should indicate GPU unavailable, got: {}", msg);
            }
        }
    }
}
```

**Run**:
```bash
cargo test --lib unified_gpu_buffer_pool_basics --lib
```

---

## Step 7: Implement Shared Component Integration (Next Sessions)

### For Each Component:

1. **SharedAttentionContext** (`src/domain/layers/components/attention_context.rs`)
   ```rust
   impl GpuComponent for SharedAttentionContext {
       fn attach_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) -> Result<()> {
           self.gpu_device = Some(device);
           Ok(())
       }
       
       fn gpu_ready(&self) -> Result<()> {
           self.gpu_device.as_ref()
               .ok_or_else(|| ModelError::Backend {
                   message: "GPU device not attached".to_string(),
               })?;
           Ok(())
       }
       
       // ... other trait methods
   }
   ```

2. **SharedFeedforward** - Similar pattern
3. **TemporalMixingLayer** - Similar pattern
4. **PolyAttention** - Similar pattern

---

## Testing Checklist

After each step, verify:

```bash
# Compiles (no warnings about unused imports)
cargo check --lib 2>&1 | grep -E "(error|warning)" || echo "✓ Clean"

# Tests pass
cargo test --lib unified_gpu_buffer_pool_basics -- --nocapture

# No clippy warnings
cargo clippy --lib 2>&1 | grep -E "warning" || echo "✓ Clippy clean"
```

---

## Common Errors & Fixes

### Error: `cannot find GpuDevice in scope`
**Fix**: Link it from `compute::gpu_device` module. For now, use placeholder and update when module is available.

### Error: `impl GpuComponent for SharedAttentionContext` conflicts
**Fix**: Ensure only one impl per type. Use blanket impl carefully.

### Error: `type mismatch in Arc<Mutex<GpuDevice>>`
**Fix**: Wrap device in `Arc<Mutex<>>` consistently. Use `Arc::clone()` to copy references.

### Test: `build timeout`
**Fix**: Kill cargo with Ctrl+C after 10 seconds. Check for infinite loops.

---

## Next: See Phase 5.5 Roadmap

Once you have UnifiedGpuBufferPool + GpuComponent trait working:

1. Head to `PHASE5.5_EXECUTION_ROADMAP.md` for full timeline
2. Follow Session 2 for component migration
3. Follow Session 3 for block implementations
4. Follow Session 4 for cleanup

---

## File Dependencies

```
src/domain/compute/
├── unified_gpu_buffer_pool.rs (CREATE)
│   └── depends on: gpu_memory.rs, gpu_device.rs
├── gpu_component.rs (CREATE)
│   └── depends on: unified_gpu_buffer_pool.rs
└── mod.rs (MODIFY to export)

src/domain/layers/components/
├── attention_context.rs (MODIFY - impl GpuComponent)
├── feedforward.rs (MODIFY - impl GpuComponent)
├── temporal_processing.rs (MODIFY - impl GpuComponent)
└── shared_gpu_manager.rs (MODIFY - add deprecation note)

tests/
└── unified_gpu_buffer_pool_basics.rs (CREATE for first test)
```

---

## Key Principles to Remember

1. **Strict No-Fallback**: If GPU isn't available, return `Err`, don't silently use CPU
2. **Arc<Mutex<>>**: GPU device is shared + mutable, always wrap this way
3. **Result Type**: All GPU operations return `Result<T>` for error propagation
4. **BufferSpec Power-of-2**: Align allocations to next power-of-2 for efficiency
5. **LRU Cache**: Reuse buffers before allocating new ones

---

## Questions to Answer as You Code

- [ ] Does `auto_detect()` error clearly if no GPU?
- [ ] Can components attach GPU device from pool?
- [ ] Does buffer cache reuse allocations?
- [ ] Do all GPU methods return `Result<T>`?
- [ ] Are capacity changes reflected in pool state?

---

## Resources

- **Full Phase Plan**: `CONSOLIDATION_GPU_IMPLEMENTATION_PHASE5.5.md`
- **Execution Timeline**: `PHASE5.5_EXECUTION_ROADMAP.md`
- **Build Commands**: `AGENTS.md` (Build & Test section)
- **GPU Status**: `GPU_BACKEND_IMPLEMENTATION_STATUS.md`
- **Previous Session**: `CONSOLIDATION_GPU_BACKEND_SESSION_SUMMARY.md`

Start with Step 1-3 and come back here when you hit questions. Good luck!
