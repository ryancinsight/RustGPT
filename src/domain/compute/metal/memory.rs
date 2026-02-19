//! Metal Memory Pool Implementation (macOS)
//!
//! Manages GPU memory allocation using Metal API.

use crate::common::errors::{ModelError, Result};
#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
use crate::domain::compute::gpu_memory::{GpuBuffer, GpuMemoryPool, MemoryStats};
#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
use std::collections::HashMap;

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
use metal::{Buffer, Device, MTLResourceOptions};

/// Metal memory pool for Apple GPU memory management
#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
#[derive(Debug)]
pub struct MetalMemoryPool {
    device: Device,
    buffers: HashMap<u64, Buffer>,
    next_id: u64,
    total_bytes: usize,
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
impl MetalMemoryPool {
    /// Create a new Metal memory pool
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| ModelError::Backend {
            message: "No Metal device available".to_string(),
        })?;

        Ok(Self {
            device,
            buffers: HashMap::new(),
            next_id: 1,
            total_bytes: 0,
        })
    }

    /// Get the underlying Metal device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get a buffer by ID
    pub fn get_buffer(&self, id: u64) -> Option<&Buffer> {
        self.buffers.get(&id)
    }

    /// Upload host data into an existing Metal buffer.
    pub fn upload_into_buffer(&mut self, cpu_data: &[f32], gpu_buffer: &GpuBuffer) -> Result<()> {
        if cpu_data.len() > gpu_buffer.size_f32() {
            return Err(ModelError::Backend {
                message: format!(
                    "Metal upload exceeds buffer capacity: data={} f32, buffer={} f32",
                    cpu_data.len(),
                    gpu_buffer.size_f32()
                ),
            });
        }

        let buffer = self
            .buffers
            .get(&gpu_buffer.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Metal upload failed: unknown buffer id {}", gpu_buffer.id),
            })?;
        let byte_len = cpu_data.len() * std::mem::size_of::<f32>();
        if byte_len > buffer.length() as usize {
            return Err(ModelError::Backend {
                message: format!(
                    "Metal upload exceeds actual buffer size: bytes={}, capacity={}",
                    byte_len,
                    buffer.length()
                ),
            });
        }

        // Shared-storage Metal buffers are host-visible; memcpy is sufficient for upload.
        unsafe {
            std::ptr::copy_nonoverlapping(
                cpu_data.as_ptr() as *const u8,
                buffer.contents() as *mut u8,
                byte_len,
            );
        }
        Ok(())
    }

    /// Download Metal buffer contents into host memory.
    pub fn download_from_buffer(&self, gpu_buffer: &GpuBuffer, cpu_data: &mut [f32]) -> Result<()> {
        if cpu_data.len() > gpu_buffer.size_f32() {
            return Err(ModelError::Backend {
                message: format!(
                    "Metal download exceeds buffer capacity: requested={} f32, buffer={} f32",
                    cpu_data.len(),
                    gpu_buffer.size_f32()
                ),
            });
        }

        let buffer = self
            .buffers
            .get(&gpu_buffer.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Metal download failed: unknown buffer id {}", gpu_buffer.id),
            })?;
        let byte_len = cpu_data.len() * std::mem::size_of::<f32>();
        if byte_len > buffer.length() as usize {
            return Err(ModelError::Backend {
                message: format!(
                    "Metal download exceeds actual buffer size: bytes={}, capacity={}",
                    byte_len,
                    buffer.length()
                ),
            });
        }

        // Shared-storage Metal buffers are host-visible; memcpy is sufficient for download.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.contents() as *const u8,
                cpu_data.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
        Ok(())
    }

    /// Copy buffer contents between Metal buffers.
    pub fn copy_between_buffers(
        &mut self,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        size: usize,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        if size > src.size_f32() || size > dst.size_f32() {
            return Err(ModelError::Backend {
                message: format!(
                    "Metal copy out of bounds: size={} f32, src={} f32, dst={} f32",
                    size,
                    src.size_f32(),
                    dst.size_f32()
                ),
            });
        }
        if src.id == dst.id {
            return Ok(());
        }

        let src_buffer = self
            .buffers
            .get(&src.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Metal copy failed: unknown src buffer id {}", src.id),
            })?;
        let dst_buffer = self
            .buffers
            .get(&dst.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Metal copy failed: unknown dst buffer id {}", dst.id),
            })?;
        let byte_len = size * std::mem::size_of::<f32>();

        // Shared-storage buffers allow host-side memcpy for device-to-device copies.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_buffer.contents() as *const u8,
                dst_buffer.contents() as *mut u8,
                byte_len,
            );
        }
        Ok(())
    }
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
impl GpuMemoryPool for MetalMemoryPool {
    fn allocate(&mut self, size_bytes: usize) -> Result<GpuBuffer> {
        let buffer = self.device.new_buffer(
            size_bytes as u64,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        let id = self.next_id;
        self.next_id += 1;
        self.total_bytes += size_bytes;
        self.buffers.insert(id, buffer);

        Ok(GpuBuffer { id, size_bytes })
    }

    fn upload(&mut self, data: &[f32]) -> Result<GpuBuffer> {
        let size_bytes = data.len() * std::mem::size_of::<f32>();
        let buffer = self.allocate(size_bytes)?;
        self.upload_into_buffer(data, &buffer)?;
        Ok(buffer)
    }

    fn download(&mut self, buffer: &GpuBuffer, output: &mut [f32]) -> Result<()> {
        self.download_from_buffer(buffer, output)
    }

    fn deallocate(&mut self, buffer: GpuBuffer) {
        if self.buffers.remove(&buffer.id).is_some() {
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// Non-metal fallback
#[cfg(not(all(feature = "gpu-metal", target_os = "macos")))]
#[derive(Debug)]
pub struct MetalMemoryPool;

#[cfg(not(all(feature = "gpu-metal", target_os = "macos")))]
impl MetalMemoryPool {
    pub fn new() -> Result<Self> {
        Err(ModelError::Backend {
            message: "Metal feature not enabled. Compile with --features gpu-metal".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
    fn metal_pool_creation() {
        if let Ok(pool) = MetalMemoryPool::new() {
            let stats = pool.memory_stats();
            assert_eq!(stats.allocation_count, 0);
        }
    }
}
