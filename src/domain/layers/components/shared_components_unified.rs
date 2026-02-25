//! Unified Shared Component Backend with GPU Auto-Detection
//!
//! This module provides a unified interface for routing shared components
//! (Attention, Feedforward, Temporal Processing) through either GPU or CPU backends
//! with automatic GPU detection and strict no-fallback semantics for troubleshooting.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  SharedComponentBackend (Auto-GPU or CPU-only)          │
//! ├─────────────────────────────────────────────────────────┤
//! │                                                           │
//! │  ┌──────────────────┐          ┌──────────────────┐     │
//! │  │ GPU Path         │          │ CPU Path         │     │
//! │  │ (Unified)        │          │ (Fallback only)  │     │
//! │  └──────────────────┘          └──────────────────┘     │
//! │         │                               │                │
//! │         ├─ Auto-Detection ─┬            │                │
//! │         │  (CUDA > Metal    │            │                │
//! │         │   > Vulkan >      │            │                │
//! │         │   WGPU)           │            │                │
//! │         │                   │            │                │
//! │         ▼                   ▼            ▼                │
//! │  ┌─────────────────────────────────────────────────┐    │
//! │  │ forward_attention()                             │    │
//! │  │ forward_feedforward()                           │    │
//! │  │ forward_temporal()                              │    │
//! │  │ backward_*() [placeholder for Phase 5.11]       │    │
//! │  └─────────────────────────────────────────────────┘    │
//! │                                                           │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Strict No-Fallback Guarantee
//!
//! When `SharedComponentBackend::auto_gpu()` is used:
//! - If GPU is detected and compiled in: GPU operations are used exclusively
//! - If GPU is detected but NOT compiled in: Error is returned (no silent CPU fallback)
//! - If no GPU is detected: Error is returned (strict GPU-only mode)
//!
//! This ensures:
//! 1. **Predictable Performance**: No unexpected CPU fallbacks during troubleshooting
//! 2. **Clear Error Messages**: Immediately identifies GPU build or availability issues
//! 3. **Testability**: GPU implementations can be validated without CPU interference

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use ndarray::{Array1, Array2};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::common::errors::ModelError;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::common::errors::Result;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute_backend::{
    ComputeBackend, resolve_compute_backend_strict_auto_gpu,
    resolve_compute_backend_strict_auto_npu,
};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::unified_gpu_backend::{
    GpuActivation, GpuTemporalType, UnifiedGpuBackend,
};

// ============================================================================
// Shared Component Backend Abstraction
// ============================================================================

/// Unified backend for shared components with automatic GPU detection.
///
/// Routes all shared component operations (Attention, Feedforward, Temporal)
/// through either GPU (with strict auto-detection) or CPU.
#[derive(Debug)]
pub enum SharedComponentBackend {
    /// GPU path with unified GPU backend (CUDA > Metal > Vulkan)
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    Gpu(Arc<Mutex<UnifiedGpuBackend>>),

    /// CPU-only path (for fallback testing only, not used in strict mode)
    Cpu,
}

impl SharedComponentBackend {
    /// Create backend with automatic GPU detection (strict mode).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No GPU is detected at runtime
    /// - A GPU is detected but matching feature flags are not compiled in
    ///
    /// # No Fallback Guarantee
    ///
    /// This function will NEVER fall back to CPU. It returns an error instead.
    /// Use this when you require GPU execution for correctness or performance testing.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn auto_gpu() -> Result<Self> {
        let backend = resolve_compute_backend_strict_auto_gpu()?;
        match backend {
            ComputeBackend::Cpu => {
                unreachable!(
                    "resolve_compute_backend_strict_auto_gpu() should never return CPU; \
                     this is a bug in compute_backend.rs"
                )
            }
            ComputeBackend::Cuda
            | ComputeBackend::Metal
            | ComputeBackend::Vulkan
            | ComputeBackend::Npu => {
                let gpu_backend = Arc::new(Mutex::new(UnifiedGpuBackend::new(backend)?));
                Ok(SharedComponentBackend::Gpu(gpu_backend))
            }
        }
    }

    /// Create backend with strict Intel NPU detection (no fallback).
    ///
    /// Returns an error unless an Intel NPU-capable adapter is available.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn auto_npu() -> Result<Self> {
        let backend = resolve_compute_backend_strict_auto_npu()?;
        let gpu_backend = Arc::new(Mutex::new(UnifiedGpuBackend::new(backend)?));
        Ok(SharedComponentBackend::Gpu(gpu_backend))
    }

    /// Create CPU-only backend (for testing fallback paths).
    ///
    /// This intentionally disables GPU acceleration and routes all operations
    /// through CPU implementations. Use only for:
    /// - Correctness validation (CPU as ground truth)
    /// - Testing fallback mechanisms
    /// - Systems where GPU is deliberately unavailable
    #[allow(dead_code)]
    pub fn cpu_only() -> Self {
        SharedComponentBackend::Cpu
    }

    /// Check if backend is GPU-based.
    pub fn is_gpu(&self) -> bool {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            matches!(self, SharedComponentBackend::Gpu(_))
        }

        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        {
            false
        }
    }

    /// Enforce GPU when available (panic if CPU selected for GPU-only ops).
    pub fn require_gpu_operation(&self, op_name: &str) {
        match self {
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            SharedComponentBackend::Gpu(_) => {
                // OK - GPU path will execute
            }
            SharedComponentBackend::Cpu => {
                panic!(
                    "GPU operation '{}' requested but SharedComponentBackend::Cpu was selected. \
                     Use SharedComponentBackend::auto_gpu() for strict GPU-only execution.",
                    op_name
                );
            }
        }
    }

    /// Enforce CPU (panic if GPU selected for CPU-only paths).
    #[allow(dead_code)]
    pub fn require_cpu_operation(&self, #[allow(unused_variables)] op_name: &str) {
        match self {
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            SharedComponentBackend::Gpu(_) => {
                panic!(
                    "CPU operation '{}' requested but GPU backend was selected. \
                     No automatic CPU fallback is allowed once a GPU backend is selected.",
                    op_name
                );
            }
            SharedComponentBackend::Cpu => {
                // OK - CPU path will execute
            }
        }
    }
}

// ============================================================================
// Backend Access Helpers (for component implementations)
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl SharedComponentBackend {
    /// Get GPU backend reference (panics if CPU path is selected).
    pub fn gpu(&self) -> &Arc<Mutex<UnifiedGpuBackend>> {
        match self {
            SharedComponentBackend::Gpu(backend) => backend,
            SharedComponentBackend::Cpu => {
                panic!("Attempted to access GPU backend in CPU-only mode")
            }
        }
    }

    #[inline]
    fn with_gpu_backend<T, F>(&self, op_name: &str, f: F) -> Result<T>
    where
        F: FnOnce(&mut UnifiedGpuBackend) -> Result<T>,
    {
        self.require_gpu_operation(op_name);
        let mut backend = self.gpu().lock().map_err(|_| ModelError::Backend {
            message: format!("Failed to lock unified GPU backend for '{}'", op_name),
        })?;
        f(&mut backend)
    }

    /// Attempt to forward attention through GPU.
    pub fn forward_attention_gpu(
        &self,
        _input: &Array2<f32>,
        _num_heads: usize,
    ) -> Result<Array2<f32>> {
        self.require_gpu_operation("forward_attention");
        Err(ModelError::Backend {
            message: "forward_attention_gpu(input, num_heads) is deprecated in SharedComponentBackend. \
                      Use forward_attention_context_gpu(...) or component-specific attention GPU dispatch."
                .to_string(),
        })
    }

    /// Attempt to forward feedforward through GPU.
    pub fn forward_feedforward_gpu(
        &self,
        _input: &Array2<f32>,
        _hidden_dim: usize,
    ) -> Result<Array2<f32>> {
        self.require_gpu_operation("forward_feedforward");
        Err(ModelError::Backend {
            message: "forward_feedforward_gpu(input, hidden_dim) is deprecated in SharedComponentBackend. \
                      Use forward_feedforward_gpu_with_weights(...) for explicit FFN dispatch."
                .to_string(),
        })
    }

    /// Attempt to forward temporal operations through GPU.
    pub fn forward_temporal_gpu(
        &self,
        _input: &Array2<f32>,
        _state_dim: usize,
    ) -> Result<Array2<f32>> {
        self.require_gpu_operation("forward_temporal");
        Err(ModelError::Backend {
            message:
                "forward_temporal_gpu(input, state_dim) is deprecated in SharedComponentBackend. \
                      Use forward_temporal_gpu_typed(...) with explicit GpuTemporalType."
                    .to_string(),
        })
    }

    /// Forward attention-context modulation through unified GPU backend.
    pub fn forward_attention_context_gpu(
        &self,
        input: &Array2<f32>,
        context: &Array2<f32>,
        strength: f32,
    ) -> Result<Array2<f32>> {
        self.with_gpu_backend("forward_attention_context", |backend| {
            backend.forward_attention_context(input, context, strength)
        })
        .map_err(|e| ModelError::Backend {
            message: format!("Attention context GPU forward failed: {}", e),
        })
    }

    /// Forward feedforward through unified GPU backend with explicit FFN parameters.
    pub fn forward_feedforward_gpu_with_weights(
        &self,
        input: &Array2<f32>,
        w1: &Array2<f32>,
        b1: &Array1<f32>,
        w2: &Array2<f32>,
        b2: &Array1<f32>,
        activation: GpuActivation,
    ) -> Result<Array2<f32>> {
        self.with_gpu_backend("forward_feedforward", |backend| {
            backend.forward_feedforward(input, w1, b1, w2, b2, activation)
        })
        .map_err(|e| ModelError::Backend {
            message: format!("Feedforward GPU forward failed: {}", e),
        })
    }

    /// Forward temporal processing through unified GPU backend with explicit temporal type.
    pub fn forward_temporal_gpu_typed(
        &self,
        input: &Array2<f32>,
        temporal_type: GpuTemporalType,
    ) -> Result<Array2<f32>> {
        self.with_gpu_backend("forward_temporal", |backend| {
            backend.forward_temporal(input, temporal_type)
        })
        .map_err(|e| ModelError::Backend {
            message: format!("Temporal GPU forward failed: {}", e),
        })
    }

    /// Backward temporal processing through unified GPU backend with explicit temporal type.
    ///
    /// Returns `(input_grads, param_grads)` where param grad ordering is backend-defined.
    pub fn backward_temporal_gpu_typed(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
        temporal_type: GpuTemporalType,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        self.with_gpu_backend("backward_temporal", |backend| {
            backend.backward_temporal(input, output_grads, temporal_type)
        })
        .map_err(|e| ModelError::Backend {
            message: format!("Temporal GPU backward failed: {}", e),
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = SharedComponentBackend::cpu_only();
        assert!(!backend.is_gpu());
    }

    #[test]
    fn test_cpu_backend_require_cpu_operation() {
        let backend = SharedComponentBackend::cpu_only();
        // Should not panic
        backend.require_cpu_operation("test_cpu_op");
    }

    #[test]
    #[should_panic(expected = "GPU operation")]
    fn test_cpu_backend_require_gpu_operation_panics() {
        let backend = SharedComponentBackend::cpu_only();
        backend.require_gpu_operation("test_gpu_op");
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_auto_gpu_resolution_on_gpu_systems() {
        // This test only runs on systems with GPU feature compilation
        // Skip test if no GPU backend is available at compile time
        match SharedComponentBackend::auto_gpu() {
            Ok(backend) => {
                assert!(backend.is_gpu(), "auto_gpu() should return GPU backend");
            }
            Err(e) => {
                // Expected if no GPU is available at runtime
                eprintln!("GPU not available: {}", e);
            }
        }
    }

    #[test]
    fn test_cpu_backend_is_not_gpu() {
        let backend = SharedComponentBackend::cpu_only();
        assert!(!backend.is_gpu());
    }
}
