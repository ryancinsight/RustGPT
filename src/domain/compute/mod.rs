//! GPU compute backend module
//!
//! Provides abstractions for GPU-accelerated computation across different backends
//! (CUDA, Metal, WebGPU). Includes device memory management, matrix operations,
//! and data transfer utilities.
//!
//! ## Unified GPU Component Trait (Phase 5.4)
//!
//! The `gpu_component` module provides a unified trait for all GPU-capable shared
//! components. All shared components implement `GpuComponent` for consistent GPU
//! management with strict no-fallback semantics.
//!
//! ## Unified GPU Executor (Phase 5.3)
//!
//! The `unified_gpu_executor` module provides a single entry point for GPU kernel
//! dispatch across all shared layer components:
//! - `SharedAttentionContext`: Context modulation, similarity computation
//! - `SharedFeedforward`: RichardsGlu, MoE forward passes
//! - `SharedTemporalProcessing`: Attention, SSM operations
//!
//! ## Unified GPU Buffer Pool (Phase 5.3)
//!
//! The `unified_gpu_buffer_pool` module provides centralized GPU memory management
//! with power-of-2 sizing and zero-allocation reuse patterns.

pub mod compute_tensor;
pub mod cross_architecture_pool;
pub mod fused_kernels;
pub mod gpu_auto_detect;
pub mod gpu_component;
pub mod gpu_device;
pub mod gpu_memory;
pub mod gpu_ops;
pub mod gpu_reduction_kernels;
pub mod gpu_richards_derivative_kernel;
pub mod gpu_softmax_kernel;
pub mod richards_glu_fused_kernel;
pub mod shared_gpu_memory_pool;
pub mod unified_gpu_buffer_pool;
pub mod unified_gpu_executor;
pub mod wgsl_kernels;

#[cfg(feature = "gpu-cuda")]
pub mod cuda;

#[cfg(feature = "gpu-metal")]
pub mod metal;

#[cfg(feature = "wgpu")]
pub mod wgpu_ops;

pub use compute_tensor::{ComputeTensor, ComputeTensor1D, TensorShape};
pub use gpu_device::GpuDevice;
pub use gpu_memory::{GpuBuffer, GpuMemoryPool, MemoryStats};
pub use gpu_ops::{GpuMatrixOps, RichardsCurveParams};
pub use gpu_reduction_kernels::GpuReductionKernel;
pub use gpu_richards_derivative_kernel::GpuRichardsDerivativeKernel;
pub use gpu_softmax_kernel::GpuSoftmaxGradientKernel;
// Note: GpuComponent is re-exported from gpu_component for consistency
pub use cross_architecture_pool::{ArchitectureFlags, CrossArchitectureBufferPool, CrossPoolStats};
pub use fused_kernels::{
    FusedKernelMetrics, FusedKernelResult, RichardsGluFusedKernelExecutor,
    RichardsGluFusedPass1Params, RichardsGluFusedPass2Params,
};
pub use gpu_auto_detect::{
    GpuAutoDetector, GpuDetectionDiagnostics, GpuDetectionStatus, GpuFeatureSet,
};
pub use gpu_component::{GpuComponent, GpuExecutionStats, GpuStatsReporting, require_gpu_or_error};
pub use shared_gpu_memory_pool::{SharedBufferSlot, SharedGpuMemoryPool, SharedPoolStats};
pub use unified_gpu_buffer_pool::{GpuBufferId, GpuPoolStats, UnifiedGpuBufferPool};
pub use unified_gpu_executor::UnifiedGpuExecutor;

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub use cross_architecture_pool::BufferPoolIntegration;

#[cfg(feature = "gpu-cuda")]
pub use cuda::memory::CudaMemoryPool;
#[cfg(feature = "gpu-cuda")]
pub use cuda::ops::CudaMatrixOps;

#[cfg(feature = "gpu-metal")]
pub use metal::memory::MetalMemoryPool;
#[cfg(feature = "gpu-metal")]
pub use metal::ops::MetalMatrixOps;

#[cfg(any(feature = "gpu-wgpu", feature = "wgpu"))]
pub use wgpu_ops::{WgpuMatrixOps, WgpuMemoryPool};

#[cfg(any(feature = "gpu-wgpu", feature = "wgpu"))]
pub use wgpu_ops::RichardsGluFusedParams;
