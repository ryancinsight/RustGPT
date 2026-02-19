//! GPU Memory Management
//!
//! Provides memory allocation, deallocation, and lifecycle management for GPU buffers.
//! Abstracts backend-specific memory handling (CUDA, Metal, Vulkan).

use crate::common::errors::Result;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt;

/// Opaque handle to GPU-resident memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GpuBuffer {
    pub(crate) id: u64,
    pub(crate) size_bytes: usize,
}

impl GpuBuffer {
    /// Size of this buffer in bytes
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Size of this buffer as number of f32 elements
    #[inline]
    pub fn size_f32(&self) -> usize {
        self.size_bytes / std::mem::size_of::<f32>()
    }
}

impl fmt::Display for GpuBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuBuffer(id={}, size={}B)", self.id, self.size_bytes)
    }
}

/// Memory statistics for GPU device
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total device memory in bytes
    pub total_bytes: usize,
    /// Currently used memory in bytes
    pub used_bytes: usize,
    /// Available free memory in bytes
    pub free_bytes: usize,
    /// Number of allocated buffers
    pub allocation_count: u32,
}

impl MemoryStats {
    /// Memory utilization percentage (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.used_bytes as f32 / self.total_bytes as f32
        }
    }

    /// Format memory stats for display
    pub fn format_human(&self) -> String {
        let mb = |bytes: usize| bytes as f32 / (1024.0 * 1024.0);
        format!(
            "GPU Memory: {:.1} MB / {:.1} MB ({:.1}%), {} buffers",
            mb(self.used_bytes),
            mb(self.total_bytes),
            self.utilization() * 100.0,
            self.allocation_count
        )
    }
}

/// Trait for GPU memory management
///
/// Implementations abstract the differences between CUDA, Metal, and Vulkan memory APIs.
pub trait GpuMemoryPool: Send + Sync {
    /// Allocate `size_bytes` on the device
    ///
    /// # Errors
    /// Returns `ModelError::Backend` if allocation fails (out of memory, etc.)
    fn allocate(&mut self, size_bytes: usize) -> Result<GpuBuffer>;

    /// Upload data to the device
    fn upload(&mut self, data: &[f32]) -> Result<GpuBuffer>;

    /// Download data from the device to a CPU slice
    ///
    /// # Arguments
    /// * `buffer` - The GPU buffer to read from
    /// * `output` - The CPU slice to write into (length must match buffer size in elements)
    fn download(&mut self, buffer: &GpuBuffer, output: &mut [f32]) -> Result<()>;

    /// Deallocate a GPU buffer
    ///
    /// # Panics
    /// May panic if the buffer ID is invalid or already freed (depends on implementation)
    fn deallocate(&mut self, buffer: GpuBuffer);

    /// Free all allocated buffers
    fn clear(&mut self);

    /// Get current memory statistics
    fn memory_stats(&self) -> MemoryStats;

    /// Suggestion: prefer power-of-2 sizing to reduce fragmentation
    ///
    /// Returns the next power-of-2 size >= required_bytes
    fn suggest_capacity(&self, required_bytes: usize) -> usize {
        required_bytes.next_power_of_two().max(256)
    }

    /// Compact allocations if possible (backend-dependent)
    ///
    /// May be a no-op on some backends. Default implementation does nothing.
    fn compact(&mut self) {}

    /// Downcast support for backend-specific fast paths.
    fn as_any(&self) -> &dyn Any;

    /// Mutable downcast support for backend-specific fast paths.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// CPU-fallback memory pool (for testing and non-GPU builds)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMemoryPool {
    buffers: std::collections::HashMap<u64, Vec<f32>>,
    next_id: u64,
    total_bytes: usize,
}

impl Default for CpuMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuMemoryPool {
    /// Create a new CPU memory pool
    pub fn new() -> Self {
        Self {
            buffers: std::collections::HashMap::new(),
            next_id: 1,
            total_bytes: 0,
        }
    }
}

impl GpuMemoryPool for CpuMemoryPool {
    fn allocate(&mut self, size_bytes: usize) -> Result<GpuBuffer> {
        let size_f32 = size_bytes / std::mem::size_of::<f32>();
        let vec = vec![0.0f32; size_f32];
        let id = self.next_id;
        self.next_id += 1;
        self.total_bytes += size_bytes;
        self.buffers.insert(id, vec);

        Ok(GpuBuffer { id, size_bytes })
    }

    fn upload(&mut self, data: &[f32]) -> Result<GpuBuffer> {
        let size_bytes = data.len() * std::mem::size_of::<f32>();
        let vec = data.to_vec();
        let id = self.next_id;
        self.next_id += 1;
        self.total_bytes += size_bytes;
        self.buffers.insert(id, vec);

        Ok(GpuBuffer { id, size_bytes })
    }

    fn download(&mut self, buffer: &GpuBuffer, output: &mut [f32]) -> Result<()> {
        let vec = self.buffers.get(&buffer.id).ok_or_else(|| {
            crate::common::errors::ModelError::Backend {
                message: format!("Buffer not found: {}", buffer.id),
            }
        })?;

        if vec.len() != output.len() {
            return Err(crate::common::errors::ModelError::Backend {
                message: format!(
                    "Buffer size mismatch: expected {}, got {}",
                    vec.len(),
                    output.len()
                ),
            });
        }

        output.copy_from_slice(vec);
        Ok(())
    }

    fn deallocate(&mut self, buffer: GpuBuffer) {
        if let Some(_) = self.buffers.remove(&buffer.id) {
            self.total_bytes -= buffer.size_bytes;
        }
    }

    fn clear(&mut self) {
        self.buffers.clear();
        self.total_bytes = 0;
    }

    fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_bytes: self.total_bytes,
            used_bytes: self.total_bytes,
            free_bytes: 0,
            allocation_count: self.buffers.len() as u32,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_pool_allocate_deallocate() {
        let mut pool = CpuMemoryPool::new();
        let buf1 = pool.allocate(1024).unwrap();
        let buf2 = pool.allocate(2048).unwrap();

        assert_eq!(buf1.size_bytes, 1024);
        assert_eq!(buf2.size_bytes, 2048);
        assert_ne!(buf1.id, buf2.id);

        let stats = pool.memory_stats();
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.used_bytes, 3072);

        pool.deallocate(buf1);
        let stats = pool.memory_stats();
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.used_bytes, 2048);
    }

    #[test]
    fn memory_stats_utilization() {
        let stats = MemoryStats {
            total_bytes: 1024,
            used_bytes: 512,
            free_bytes: 512,
            allocation_count: 1,
        };
        assert_eq!(stats.utilization(), 0.5);
    }

    #[test]
    fn suggest_capacity_power_of_two() {
        let pool = CpuMemoryPool::new();
        // 100 → 256 (next power of 2 is 128, but min is 256)
        assert_eq!(pool.suggest_capacity(100), 256);
        // 256 → 256 (already power of 2)
        assert_eq!(pool.suggest_capacity(256), 256);
        // 300 → 512 (next power of 2 after 256)
        assert_eq!(pool.suggest_capacity(300), 512);
        // 128 → 256 (next power of 2 is 128, but min is 256)
        assert_eq!(pool.suggest_capacity(128), 256);
    }
}
