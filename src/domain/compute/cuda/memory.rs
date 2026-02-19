//! CUDA Memory Management
//!
//! Provides GPU memory allocation and lifecycle management via cudarc.

use crate::common::errors::{ModelError, Result};
use crate::domain::compute::gpu_memory::{GpuBuffer, GpuMemoryPool, MemoryStats};
#[cfg(feature = "gpu-cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};
use std::collections::HashMap;
use std::sync::Arc;

/// CUDA GPU memory pool
///
/// Manages GPU-resident buffers allocated via cudarc.
/// CudaDevice::new returns Arc<CudaDevice>.
#[cfg(feature = "gpu-cuda")]
pub struct CudaMemoryPool {
    device: Arc<CudaDevice>,
    buffers: HashMap<u64, CudaSlice<f32>>,
    next_id: u64,
    total_allocated: usize,
}

#[cfg(feature = "gpu-cuda")]
impl CudaMemoryPool {
    /// Create a new CUDA memory pool for the specified device
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(ordinal).map_err(|e| ModelError::Backend {
            message: format!("Failed to initialize CUDA device {}: {}", ordinal, e),
        })?;

        Ok(Self {
            device,
            buffers: HashMap::new(),
            next_id: 1,
            total_allocated: 0,
        })
    }

    /// Get the CUDA device
    #[allow(dead_code)]
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Cloneable CUDA device handle for backend operation dispatchers.
    #[allow(dead_code)]
    pub fn device_handle(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }

    /// Upload host data into an existing CUDA buffer.
    pub fn upload_into_buffer(&mut self, cpu_data: &[f32], gpu_buffer: &GpuBuffer) -> Result<()> {
        if cpu_data.len() > gpu_buffer.size_f32() {
            return Err(ModelError::Backend {
                message: format!(
                    "CUDA upload exceeds buffer capacity: data={} f32, buffer={} f32",
                    cpu_data.len(),
                    gpu_buffer.size_f32()
                ),
            });
        }

        let slice = self
            .buffers
            .get_mut(&gpu_buffer.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("CUDA upload failed: unknown buffer id {}", gpu_buffer.id),
            })?;

        self.device
            .htod_sync_copy_into(cpu_data, slice)
            .map_err(|e| ModelError::Backend {
                message: format!("CUDA upload failed: {}", e),
            })
    }

    /// Download CUDA buffer contents into host memory.
    pub fn download_from_buffer(&self, gpu_buffer: &GpuBuffer, cpu_data: &mut [f32]) -> Result<()> {
        if cpu_data.len() > gpu_buffer.size_f32() {
            return Err(ModelError::Backend {
                message: format!(
                    "CUDA download exceeds buffer capacity: requested={} f32, buffer={} f32",
                    cpu_data.len(),
                    gpu_buffer.size_f32()
                ),
            });
        }

        let slice = self
            .buffers
            .get(&gpu_buffer.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("CUDA download failed: unknown buffer id {}", gpu_buffer.id),
            })?;

        self.device
            .dtoh_sync_copy_into(slice, cpu_data)
            .map_err(|e| ModelError::Backend {
                message: format!("CUDA download failed: {}", e),
            })
    }

    /// Copy buffer data between CUDA buffers.
    ///
    /// Note: this currently stages through host memory to keep borrowing rules simple.
    /// A direct `dtod_copy` fast path can replace this in a later kernel pass.
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
                    "CUDA copy out of bounds: size={} f32, src={} f32, dst={} f32",
                    size,
                    src.size_f32(),
                    dst.size_f32()
                ),
            });
        }
        if src.id == dst.id {
            return Ok(());
        }

        let mut staging = vec![0.0f32; size];
        {
            let src_slice = self
                .buffers
                .get(&src.id)
                .ok_or_else(|| ModelError::Backend {
                    message: format!("CUDA copy failed: unknown src buffer id {}", src.id),
                })?;
            self.device
                .dtoh_sync_copy_into(src_slice, &mut staging)
                .map_err(|e| ModelError::Backend {
                    message: format!("CUDA copy read failed: {}", e),
                })?;
        }
        {
            let dst_slice = self
                .buffers
                .get_mut(&dst.id)
                .ok_or_else(|| ModelError::Backend {
                    message: format!("CUDA copy failed: unknown dst buffer id {}", dst.id),
                })?;
            self.device
                .htod_sync_copy_into(&staging, dst_slice)
                .map_err(|e| ModelError::Backend {
                    message: format!("CUDA copy write failed: {}", e),
                })?;
        }
        Ok(())
    }
}

#[cfg(feature = "gpu-cuda")]
impl Default for CudaMemoryPool {
    fn default() -> Self {
        // Default to device 0
        Self::new(0).unwrap_or_else(|_| panic!("Failed to initialize default CUDA device 0"))
    }
}

#[cfg(feature = "gpu-cuda")]
impl GpuMemoryPool for CudaMemoryPool {
    fn allocate(&mut self, size_bytes: usize) -> Result<GpuBuffer> {
        if size_bytes == 0 {
            return Err(ModelError::Backend {
                message: "CUDA: Cannot allocate 0 bytes".to_string(),
            });
        }

        let elem_size = std::mem::size_of::<f32>();
        if size_bytes % elem_size != 0 {
            return Err(ModelError::Backend {
                message: format!(
                    "CUDA: allocation size {} is not aligned to f32 element size {}",
                    size_bytes, elem_size
                ),
            });
        }
        let size_f32 = size_bytes / elem_size;
        let slice = self
            .device
            .alloc_zeros::<f32>(size_f32)
            .map_err(|e| ModelError::Backend {
                message: format!("CUDA allocation failed: {}", e),
            })?;

        let id = self.next_id;
        self.next_id += 1;
        self.total_allocated += size_bytes;
        self.buffers.insert(id, slice);

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
        if let Some(_) = self.buffers.remove(&buffer.id) {
            self.total_allocated = self.total_allocated.saturating_sub(buffer.size_bytes);
        }
    }

    fn clear(&mut self) {
        self.buffers.clear();
        self.total_allocated = 0;
    }

    fn memory_stats(&self) -> MemoryStats {
        // TODO: Query device total memory via cudarc
        let device_total = 8 * 1024 * 1024 * 1024; // Assume 8 GB for now

        MemoryStats {
            total_bytes: device_total,
            used_bytes: self.total_allocated,
            free_bytes: device_total.saturating_sub(self.total_allocated),
            allocation_count: self.buffers.len() as u32,
        }
    }

    fn compact(&mut self) {
        // CUDA memory fragmentation is handled by the driver
        // This is a no-op on CUDA
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
