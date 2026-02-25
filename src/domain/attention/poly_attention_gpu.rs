//! GPU integration helpers for `PolyAttention`.
//!
//! This module provides GPU context setup and helper functions for PolyAttention paths.
//! The actual GPU computation is handled by the unified GPU kernels in the components layer.
//!
//! ## Architecture
//!
//! PolyAttention GPU support is provided through multiple layers:
//! - `poly_attention_gpu.rs` (this file): GPU context setup and helpers
//! - `attention_gpu_kernel.rs`: Core GPU attention kernel
//! - `unified_gpu_kernels.rs`: Fused computation kernels
//! - `gpu_backend_variants.rs`: Backend dispatch
//!
//! ## Usage
//!
//! GPU is enabled automatically when using `ComputeBackendPreference::AutoGpu` or
//! explicitly via `ComputeBackendPreference::Npu` for NPU-only execution.

use std::sync::{Arc, Mutex};

use crate::common::errors::{ModelError, Result};
use crate::domain::compute::GpuDevice;
use crate::domain::compute_backend::ComputeBackend;

/// GPU context for PolyAttention operations.
///
/// This struct holds the GPU device and provides methods for
/// managing GPU resources for attention computation.
#[derive(Debug)]
pub struct PolyAttentionGpuContext {
    /// The GPU device for computation
    device: Arc<Mutex<GpuDevice>>,
    /// Backend name for logging
    backend_name: &'static str,
    /// Whether the device is an Intel NPU
    is_npu: bool,
}

impl PolyAttentionGpuContext {
    /// Create a new GPU context with auto-detection.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU is available.
    pub fn new() -> Result<Self> {
        let device = GpuDevice::auto_detect()?;
        let backend_name = device.backend_name();
        
        // Check if using NPU
        #[cfg(feature = "wgpu")]
        let is_npu = device.wgpu_device()
            .map(|_| {
                // The WgpuMemoryPool tracks whether it's an NPU
                false // Simplified - actual detection happens in WgpuMemoryPool
            })
            .unwrap_or(false);
        
        #[cfg(not(feature = "wgpu"))]
        let is_npu = false;

        tracing::info!(
            backend = backend_name,
            npu = is_npu,
            "PolyAttention GPU context initialized"
        );

        Ok(Self {
            device: Arc::new(Mutex::new(device)),
            backend_name,
            is_npu,
        })
    }

    /// Create a new GPU context targeting NPU specifically.
    ///
    /// # Errors
    ///
    /// Returns an error if no NPU is available.
    pub fn new_npu() -> Result<Self> {
        let device = GpuDevice::auto_detect_npu()?;
        let backend_name = device.backend_name();
        
        tracing::info!(
            backend = backend_name,
            "PolyAttention NPU context initialized"
        );

        Ok(Self {
            device: Arc::new(Mutex::new(device)),
            backend_name: "npu",
            is_npu: true,
        })
    }

    /// Get the GPU device.
    pub fn device(&self) -> Arc<Mutex<GpuDevice>> {
        Arc::clone(&self.device)
    }

    /// Get the backend name.
    pub fn backend_name(&self) -> &'static str {
        self.backend_name
    }

    /// Check if this context is using an NPU.
    pub fn is_npu(&self) -> bool {
        self.is_npu
    }

    /// Format device info for display.
    pub fn format_info(&self) -> String {
        let device = self.device.lock().unwrap();
        device.format_info()
    }
}

impl Default for PolyAttentionGpuContext {
    fn default() -> Self {
        Self::new().expect("Failed to create default GPU context")
    }
}

/// Configuration for GPU attention execution.
#[derive(Debug, Clone)]
pub struct GpuAttentionConfig {
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Scale factor (1 / sqrt(head_dim))
    pub scale: f32,
    /// Whether to use causal masking
    pub causal: bool,
    /// Optional window size for sliding window attention
    pub window_size: Option<usize>,
}

impl GpuAttentionConfig {
    /// Create a new GPU attention config.
    pub fn new(
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        embed_dim: usize,
    ) -> Self {
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Self {
            batch_size,
            seq_len,
            num_heads,
            embed_dim,
            head_dim,
            scale,
            causal: true,
            window_size: None,
        }
    }

    /// Set causal masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set window size for sliding window attention.
    pub fn with_window_size(mut self, window_size: Option<usize>) -> Self {
        self.window_size = window_size;
        self
    }
}

/// Detect and return information about available GPU/NPU backends.
#[derive(Debug, Clone)]
pub struct GpuBackendInfo {
    /// Whether CUDA is available
    pub cuda_available: bool,
    /// Whether Metal is available
    pub metal_available: bool,
    /// Whether Vulkan/WGPU is available
    pub vulkan_available: bool,
    /// Whether an Intel NPU is available
    pub npu_available: bool,
    /// The recommended backend for auto-detection
    pub recommended_backend: ComputeBackend,
}

impl GpuBackendInfo {
    /// Detect available GPU backends.
    ///
    /// Note: This is a best-effort detection. Actual availability
    /// may differ based on drivers and runtime conditions.
    pub fn detect() -> Self {
        use crate::domain::compute_backend::detect_available_gpu_backends;

        let backends = detect_available_gpu_backends();
        
        let cuda_available = backends.contains(&ComputeBackend::Cuda);
        let metal_available = backends.contains(&ComputeBackend::Metal);
        let vulkan_available = backends.contains(&ComputeBackend::Vulkan);
        let npu_available = backends.contains(&ComputeBackend::Npu);

        // Priority: NPU > CUDA > Vulkan > Metal
        let recommended_backend = if npu_available {
            ComputeBackend::Npu
        } else if cuda_available {
            ComputeBackend::Cuda
        } else if vulkan_available {
            ComputeBackend::Vulkan
        } else if metal_available {
            ComputeBackend::Metal
        } else {
            ComputeBackend::Cpu
        };

        Self {
            cuda_available,
            metal_available,
            vulkan_available,
            npu_available,
            recommended_backend,
        }
    }

    /// Get a human-readable summary of available backends.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        
        if self.npu_available {
            parts.push("NPU");
        }
        if self.cuda_available {
            parts.push("CUDA");
        }
        if self.vulkan_available {
            parts.push("Vulkan");
        }
        if self.metal_available {
            parts.push("Metal");
        }
        
        if parts.is_empty() {
            "None (CPU only)".to_string()
        } else {
            parts.join(", ")
        }
    }
}

impl Default for GpuBackendInfo {
    fn default() -> Self {
        Self::detect()
    }
}

/// Validate GPU configuration for attention operations.
pub fn validate_attention_config(config: &GpuAttentionConfig) -> Result<()> {
    if config.batch_size == 0 {
        return Err(ModelError::InvalidInput {
            message: "batch_size must be greater than 0".to_string(),
        });
    }
    
    if config.seq_len == 0 {
        return Err(ModelError::InvalidInput {
            message: "seq_len must be greater than 0".to_string(),
        });
    }
    
    if config.num_heads == 0 {
        return Err(ModelError::InvalidInput {
            message: "num_heads must be greater than 0".to_string(),
        });
    }
    
    if config.embed_dim == 0 {
        return Err(ModelError::InvalidInput {
            message: "embed_dim must be greater than 0".to_string(),
        });
    }
    
    if config.embed_dim % config.num_heads != 0 {
        return Err(ModelError::InvalidInput {
            message: format!(
                "embed_dim ({}) must be divisible by num_heads ({})",
                config.embed_dim, config.num_heads
            ),
        });
    }
    
    if let Some(window_size) = config.window_size {
        if window_size > config.seq_len {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "window_size ({}) must be <= seq_len ({})",
                    window_size, config.seq_len
                ),
            });
        }
    }
    
    Ok(())
}

/// Estimate memory usage for GPU attention computation.
pub fn estimate_attention_memory(config: &GpuAttentionConfig) -> usize {
    let total_tokens = config.batch_size * config.seq_len;
    let embed_dim = config.embed_dim;
    let head_dim = config.head_dim;
    
    // Memory for Q, K, V projections: 3 * batch * seq * embed * sizeof(f32)
    let qkv_memory = 3 * total_tokens * embed_dim * std::mem::size_of::<f32>();
    
    // Memory for attention scores: batch * heads * seq * seq * sizeof(f32)
    let num_heads = config.num_heads;
    let seq_len = config.seq_len;
    let scores_memory = config.batch_size * num_heads * seq_len * seq_len * std::mem::size_of::<f32>();
    
    // Memory for output projection
    let output_memory = total_tokens * embed_dim * std::mem::size_of::<f32>();
    
    // Estimate 20% overhead for workspace buffers
    let overhead = (qkv_memory + scores_memory + output_memory) / 5;
    
    qkv_memory + scores_memory + output_memory + overhead
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_detection() {
        let info = GpuBackendInfo::detect();
        println!("Available GPU backends: {}", info.summary());
        println!("Recommended backend: {:?}", info.recommended_backend);
    }

    #[test]
    fn test_attention_config_validation() {
        // Valid config
        let config = GpuAttentionConfig::new(2, 512, 8, 512);
        assert!(validate_attention_config(&config).is_ok());

        // Invalid: embed_dim not divisible by num_heads
        let mut config = GpuAttentionConfig::new(2, 512, 8, 500);
        assert!(validate_attention_config(&config).is_err());

        // Invalid: window_size > seq_len
        config = GpuAttentionConfig::new(2, 512, 8, 512);
        config.window_size = Some(1024);
        assert!(validate_attention_config(&config).is_err());
    }

    #[test]
    fn test_memory_estimation() {
        let config = GpuAttentionConfig::new(2, 512, 8, 512);
        let memory = estimate_attention_memory(&config);
        println!("Estimated GPU memory: {} bytes ({:.2} MB)", 
            memory, 
            memory as f64 / (1024.0 * 1024.0));
        assert!(memory > 0);
    }

    #[test]
    fn test_gpu_context_creation() {
        match PolyAttentionGpuContext::new() {
            Ok(ctx) => {
                println!("GPU context created: {}", ctx.format_info());
                println!("Backend: {}", ctx.backend_name());
                println!("Is NPU: {}", ctx.is_npu());
            }
            Err(e) => {
                println!("No GPU available (expected on systems without GPU): {}", e);
            }
        }
    }
}
