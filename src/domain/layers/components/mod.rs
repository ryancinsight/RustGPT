//! Components Module
//!
//! This module contains reusable components that can be used across different architectures.
//! Shared components are designed to reduce code duplication and improve maintainability.
//!
//! ## Consolidated Memory Management
//!
//! The `UnifiedLayerWorkspace` provides consolidated buffer management for all block types
//! (Transformer, Diffusion, SSM). It replaces the previous separate implementations:
//! - `IntermediateBufferPool` (removed)
//! - `WorkspacePool` (removed)
//! - `FilmParameterCache` (removed - functionality integrated into conditioning module)
//!
//! Use `WorkspaceManaged` trait for consistent interface across all workspace types.
//!
//! ## GPU Backend Integration (Phase 5.6+)
//!
//! **Consolidated GPU Backend**: Use `UnifiedGpuBackend` for all GPU operations.
//! It provides automatic GPU detection with strict no-fallback semantics.
//!
//! ```ignore
//! // Automatic GPU detection (strict - errors if no GPU)
//! let backend = UnifiedGpuBackend::auto_detect()?;
//!
//! // Execute operations
//! let output = backend.forward_attention_context(&input, &context, 1.0)?;
//! ```
//!
//! **GPU Kernels**: Use `UnifiedGpuKernels` for low-level GPU kernel dispatch:
//! ```ignore
//! let kernels = UnifiedGpuKernels::auto_detect()?;
//! let output = kernels.attention_forward(&input, &wq, &wk, &wv, &wo, &params)?;
//! ```
//!
//! **GPU Backend Variants**: Use architecture-specific backends for optimized operations:
//! ```ignore
//! // Diffusion GPU backend
//! let diffusion = DiffusionGpuBackend::auto_detect(1000)?;
//! let noisy = diffusion.forward_diffusion(&clean_input, t, None)?;
//!
//! // SSM GPU backend (Mamba or RG-LRU)
//! let ssm = SsmGpuBackend::mamba(256, 512, 128, 32)?;
//! let output = ssm.forward(&input)?;
//!
//! // Transformer GPU backend
//! let transformer = TransformerGpuBackend::auto_detect(8, 512, 128, 32)?;
//! let attn_output = transformer.attention_forward(&input, &wq, &wk, &wv, &wo)?;
//! ```
//!
//! **Buffer Pool**: Use `UnifiedBufferPool` for cross-architecture memory sharing:
//! ```ignore
//! let pool = UnifiedBufferPool::new();
//! let buffer = pool.allocate(1024)?; // 1 KB buffer
//! ```
//!
//! Legacy GPU managers (`SharedComponentGpuManager`, `GpuSharedOpsContext`) are deprecated.
//! Migrate to `UnifiedGpuBackend` and `GpuComponent` trait.

pub mod adaptive_residuals;
pub mod adaptive_residuals_workspace;
pub mod attention_context;
pub mod attention_context_gpu;
pub mod attention_gpu_kernel;
pub mod block_core;
pub mod common;
pub mod conditioning;
pub mod feedforward;
pub mod feedforward_gpu;
pub mod fused_kernels_module;
pub mod gpu_backward_fusion;
pub mod gpu_gemm_kernels;
pub mod gpu_shared_executor;
pub mod gradient_router;
pub mod shared_gpu_manager;
pub mod ssm_gpu_kernels;
pub mod temporal_processing;
pub mod temporal_processing_gpu;
pub mod unified_buffer_pool;
pub mod unified_gpu_backend;
pub mod unified_gpu_kernels;
pub mod unified_layer_workspace;
pub mod workspace_managed;

// GPU Backend Variants (Phase 5.6)
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub mod gpu_backend_variants;

// Re-export commonly used components for convenient access
pub use adaptive_residuals_workspace::AdaptiveResidualsWorkspace;
pub use attention_context::SharedAttentionContext;
pub use fused_kernels_module::{
    attention_context_ops, mamba_scan_kernel, poly_attention_fused, richards_glu_fused,
};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub use gpu_shared_executor::{GpuExecutionStats, GpuSharedExecutor};
pub use shared_gpu_manager::SharedComponentGpuManager;
pub use unified_buffer_pool::{
    BufferHandle, BufferPoolConfig, BufferPoolStats, SharedBufferManager, UnifiedBufferPool,
};
pub use unified_gpu_backend::{GpuActivation, GpuBackendStats, GpuTemporalType, UnifiedGpuBackend};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub use unified_gpu_kernels::{AttentionParams, NormParams, SsmParams, UnifiedGpuKernels};
pub use unified_layer_workspace::UnifiedLayerWorkspace;
pub use workspace_managed::{StreamingWorkspaceManaged, WorkspaceManaged, WorkspaceStats};

// Re-export GPU backend variants
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub use gpu_backend_variants::{
    DiffusionGpuBackend, GpuBackendFactory, MoeGpuBackend, MoeParams, NoiseScheduleParams,
    NoiseScheduleType, SsmGpuBackend, TransformerGpuBackend,
};
