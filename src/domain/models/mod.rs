pub mod builder;
pub mod config;
pub mod llm;
pub mod persistence;
#[path = "titans.rs"]
pub mod titans;

// GPU-native model implementations
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub mod gpu_kv_cache;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub mod gpu_llm;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub use gpu_kv_cache::{GpuKVCache, GpuKVCacheConfig, GpuKVCacheLayer, GpuKVCacheManager};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub use gpu_llm::{
    GpuLLMModel, GpuLayer, GpuTransformerLayer, GpuSSMLayer, GpuMoELayer, GpuModelWorkspace,
};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub use crate::domain::layers::components::unified_gpu_backend::GpuActivation;
