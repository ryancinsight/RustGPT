//! CUDA Matrix Operations Implementation
//!
//! Provides GPU-accelerated matrix operations using cuBLAS via cudarc.

use crate::common::errors::{ModelError, Result};
use crate::domain::compute::gpu_memory::{GpuBuffer, GpuMemoryPool};
use crate::domain::compute::gpu_ops::GpuMatrixOps;
use std::sync::Arc;

#[cfg(feature = "gpu-cuda")]
use cudarc::driver::CudaDevice;

/// CUDA matrix operations using cuBLAS
#[cfg(feature = "gpu-cuda")]
#[derive(Debug)]
pub struct CudaMatrixOps {
    device: Arc<CudaDevice>,
}

#[cfg(feature = "gpu-cuda")]
impl CudaMatrixOps {
    /// Create new CUDA matrix operations
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    /// Get the underlying CUDA device
    pub fn device(&self) -> &CudaDevice {
        self.device.as_ref()
    }
}

#[cfg(feature = "gpu-cuda")]
impl GpuMatrixOps for CudaMatrixOps {
    fn gemm_f32(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _alpha: f32,
        _a: &GpuBuffer,
        _b: &GpuBuffer,
        _beta: f32,
        _output: &mut GpuBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // cuBLAS GEMM would go here
        // For now, return not implemented
        Err(ModelError::Backend {
            message: format!(
                "CUDA GEMM not yet implemented for shape ({}, {}) x ({}, {}). \
                 Requires cuBLAS integration.",
                m, k, k, n
            ),
        })
    }

    fn gemv_f32(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _alpha: f32,
        _a: &GpuBuffer,
        _x: &GpuBuffer,
        _beta: f32,
        _output: &mut GpuBuffer,
        m: usize,
        n: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!(
                "CUDA GEMV not yet implemented for shape ({}, {}) x ({}, 1)",
                m, n, n
            ),
        })
    }

    fn relu(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!("CUDA ReLU not yet implemented for size {}", size),
        })
    }

    fn gelu(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!("CUDA GELU not yet implemented for size {}", size),
        })
    }

    fn silu(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!("CUDA SiLU not yet implemented for size {}", size),
        })
    }

    fn sigmoid(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!("CUDA Sigmoid not yet implemented for size {}", size),
        })
    }

    fn mul(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _input1: &GpuBuffer,
        _input2: &GpuBuffer,
        _output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!("CUDA mul not yet implemented for size {}", size),
        })
    }

    fn add_scaled(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _scale: f32,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!("CUDA add_scaled not yet implemented for size {}", size),
        })
    }

    fn scale(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _scale: f32,
        _output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!("CUDA scale not yet implemented for size {}", size),
        })
    }

    fn axpy(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _a: f32,
        _input1: &GpuBuffer,
        _b: f32,
        _input2: &GpuBuffer,
        _output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!("CUDA axpy not yet implemented for size {}", size),
        })
    }

    fn layer_norm(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _gamma: &GpuBuffer,
        _beta: &GpuBuffer,
        _output: &mut GpuBuffer,
        batch_size: usize,
        feature_size: usize,
        _eps: f32,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!(
                "CUDA layer_norm not yet implemented for shape ({}, {})",
                batch_size, feature_size
            ),
        })
    }

    fn softmax(
        &mut self,
        _pool: &dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!(
                "CUDA softmax not yet implemented for shape ({}, {})",
                rows, cols
            ),
        })
    }

    fn sum(&mut self, _pool: &mut dyn GpuMemoryPool, _buffer: &GpuBuffer, size: usize) -> Result<f32> {
        Err(ModelError::Backend {
            message: format!("CUDA sum not yet implemented for size {}", size),
        })
    }

    fn mean(&mut self, _pool: &mut dyn GpuMemoryPool, _buffer: &GpuBuffer, size: usize) -> Result<f32> {
        Err(ModelError::Backend {
            message: format!("CUDA mean not yet implemented for size {}", size),
        })
    }

    fn download(
        &self,
        pool: &dyn GpuMemoryPool,
        gpu_buffer: &GpuBuffer,
        cpu_data: &mut [f32],
    ) -> Result<()> {
        if let Some(cuda_pool) = pool.as_any().downcast_ref::<super::CudaMemoryPool>() {
            return cuda_pool.download_from_buffer(gpu_buffer, cpu_data);
        }
        Err(ModelError::Backend {
            message: "CUDA download failed: invalid pool type".to_string(),
        })
    }

    fn upload(
        &mut self,
        pool: &dyn GpuMemoryPool,
        cpu_data: &[f32],
        gpu_buffer: &mut GpuBuffer,
    ) -> Result<()> {
        // We need mutable access to the pool, but the trait gives us &dyn GpuMemoryPool.
        // Wait, GpuMatrixOps methods take &mut self, but pool is passed as argument.
        // If we need to mutate the pool (e.g. for upload), we might need internal mutability or the trait signature should have &mut pool.
        //
        // Looking at GpuMatrixOps definition:
        // fn upload(&mut self, pool: &dyn GpuMemoryPool, cpu_data: &[f32], gpu_buffer: &mut GpuBuffer)
        //
        // But CudaMemoryPool::upload_into_buffer takes &mut self?
        // Let's check CudaMemoryPool in cuda/memory.rs (I read it earlier).
        //
        // Yes: pub fn upload_into_buffer(&mut self, cpu_data: &[f32], gpu_buffer: &GpuBuffer) -> Result<()>
        //
        // This is a problem. The trait signature `pool: &dyn GpuMemoryPool` assumes the pool is shared/immutable for ops, but `upload` might need to mutate the pool if it manages synchronization or something?
        //
        // Actually, `CudaMemoryPool` uses `cudarc`, which likely uses interior mutability (Arc<CudaDevice>) for the device, but the `buffers` HashMap needs mutability if we are adding buffers. But here we are uploading *into* an existing buffer.
        //
        // If `upload_into_buffer` takes `&mut self`, then we can't call it with `&dyn GpuMemoryPool`.
        //
        // Let's check `cuda/memory.rs` again.
        //
        // pub fn upload_into_buffer(&mut self, cpu_data: &[f32], gpu_buffer: &GpuBuffer) -> Result<()>
        //
        // It takes `&mut self` probably because it accesses `self.buffers`.
        //
        // If I can't change `CudaMemoryPool`, I might have to use interior mutability in `CudaMemoryPool` (RwLock<HashMap>) or change the trait to `&mut dyn GpuMemoryPool`.
        //
        // I changed `GpuMatrixOps` to take `pool: &dyn GpuMemoryPool`.
        //
        // If I change the trait to `pool: &mut dyn GpuMemoryPool`, that would work for `upload`.
        // But `GpuDevice` holds `memory: Box<dyn GpuMemoryPool>`.
        // And `GpuDevice` methods take `&mut self`. So `self.memory` is available as mutable.
        //
        // So I should have defined `GpuMatrixOps` to take `&mut dyn GpuMemoryPool`.
        //
        // Let's check `gpu_ops.rs` again.

        // I haven't read `gpu_ops.rs` in this turn, but I know I changed it.
        // I probably used `&dyn GpuMemoryPool`.
        //
        // If I change it to `&mut dyn GpuMemoryPool`, I need to update all signatures again.
        //
        // Alternatively, does `upload_into_buffer` really need `&mut self`?
        // It reads from `self.buffers`. `get` takes `&self`.
        // It writes to the buffer content. `CudaSlice` handles might be immutable but point to mutable GPU memory.
        //
        // `CudaMemoryPool` in `cuda/memory.rs`:
        //
        // pub fn upload_into_buffer(&mut self, cpu_data: &[f32], gpu_buffer: &GpuBuffer) -> Result<()> {
        //     ...
        //     let buffer = self.buffers.get(&gpu_buffer.id)...
        //     ...
        // }
        //
        // It calls `self.buffers.get`. That only needs `&self`.
        // So `upload_into_buffer` could take `&self` if `cudarc` allows it.
        // `cudarc` `htod_sync_copy_into` takes `&mut self` on the slice? No, `CudaSlice` is a handle.
        //
        // Let's look at `metal/memory.rs`:
        // pub fn upload_into_buffer(&mut self, ...)
        // It uses `self.buffers.get`.
        // It uses `buffer.contents()`.
        //
        // So technically `upload_into_buffer` only needs `&self` if the buffer map is not being modified (we are not allocating).
        //
        // So the fix is to change `upload_into_buffer` (and `download`, `copy`) to take `&self` in the memory pools, instead of `&mut self`.
        //
        // `copy_between_buffers` might need `&mut self` if it needs a temporary staging buffer?
        // In `cuda/memory.rs`, `copy_between_buffers` uses `staging` vector which is local.
        //
        // So, I will change `CudaMemoryPool` and `MetalMemoryPool` methods to take `&self` where possible.
        //
        // However, I cannot modify `CudaMemoryPool` right now as I am focusing on `cuda/ops.rs`.
        //
        // Actually, I can fix `cuda/memory.rs` and `metal/memory.rs` as well.
        //
        // But wait, `copy_between_buffers` in `cuda/memory.rs` calls `self.buffers.get_mut(&dst.id)`.
        // Why `get_mut`?
        // `CudaSlice`?
        //
        // If `CudaSlice` requires mutable access to write to it, then we need `&mut self` on the pool if the pool owns the slices.
        //
        // If `GpuMatrixOps` takes `pool: &dyn GpuMemoryPool`, we are stuck.
        //
        // Maybe I should change `GpuMatrixOps` to take `pool: &mut dyn GpuMemoryPool`.
        //
        // In `GpuDevice`, we have `memory: Box<dyn GpuMemoryPool>`.
        // And `gemm_f32` takes `&mut self`.
        // So we can pass `self.memory.as_mut()`.
        //
        // So `GpuMatrixOps` SHOULD take `&mut dyn GpuMemoryPool` to be safe and allow mutation if needed (e.g. if we wanted to allocate temp buffers in the pool during an op).
        //
        // So I should refactor `GpuMatrixOps` again to use `&mut dyn GpuMemoryPool`.
        //
        // Check `gpu_ops.rs` first.

        Err(ModelError::Backend {
            message: "CUDA upload not yet implemented (requires mutable pool access refactor)"
                .to_string(),
        })
    }

    fn copy_within_device(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _src: &GpuBuffer,
        _dst: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!(
                "CUDA copy_within_device not yet implemented for size {}",
                size
            ),
        })
    }

    fn copy_within_device_range(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _src: &GpuBuffer,
        _src_offset: usize,
        _dst: &mut GpuBuffer,
        _dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!(
                "CUDA copy_within_device_range not yet implemented for size {}",
                size
            ),
        })
    }

    fn richards_curve(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _params: &crate::domain::compute::gpu_ops::RichardsCurveParams,
        size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!(
                "CUDA richards_curve not yet implemented for size {}. \
                 Use WGPU backend or compile with native CUDA kernels.",
                size
            ),
        })
    }

    fn moh_gate_activation(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _logits: &GpuBuffer,
        _alpha: &GpuBuffer,
        _beta: &GpuBuffer,
        _gate_params: &crate::domain::compute::gpu_ops::RichardsCurveParams,
        _output: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: format!(
                "CUDA moh_gate_activation not yet implemented for batch_size={}, num_heads={}",
                batch_size, num_heads
            ),
        })
    }
}

// Non-cudarc fallback
#[cfg(not(feature = "gpu-cuda"))]
#[derive(Debug)]
pub struct CudaMatrixOps;

#[cfg(not(feature = "gpu-cuda"))]
impl CudaMatrixOps {
    pub fn new() -> Result<Self> {
        Err(ModelError::Backend {
            message: "CUDA feature not enabled. Compile with --features gpu-cuda".to_string(),
        })
    }
}
