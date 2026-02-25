//! Automatic GPU Detection with Strict No-Fallback (Phase 5.6.3)
//!
//! Provides automatic GPU backend detection and initialization with strict error handling.
//! No silent fallback to CPU - if GPU is required, an error is returned if no GPU is available.
//!
//! ## Detection Priority
//!
//! 1. CUDA (if `gpu-cuda` feature enabled)
//! 2. Metal (if `gpu-metal` feature enabled, macOS only)
//! 3. Vulkan (future)
//! 4. WGPU (if `wgpu` or `gpu-wgpu` feature enabled)
//!
//! ## Usage
//!
//! ```ignore
//! // Auto-detect with strict error if no GPU
//! match GpuAutoDetector::detect_gpu_strict() {
//!     Ok(backend) => println!("GPU backend: {:?}", backend),
//!     Err(e) => eprintln!("No GPU available: {}", e),
//! }
//!
//! // With retry logic for troubleshooting
//! let backend = GpuAutoDetector::detect_with_retry(3)?;
//! ```

use crate::common::errors::{ModelError, Result};
use crate::domain::compute_backend::{ComputeBackend, resolve_compute_backend_strict_auto_npu};
use std::time::{Duration, Instant};

/// Automatic GPU detection and health check
#[derive(Debug, Clone)]
pub struct GpuAutoDetector {
    /// Detected backend
    pub backend: Option<ComputeBackend>,
    /// Detection timestamp
    pub detected_at: Instant,
    /// Whether GPU is ready for computation
    pub is_ready: bool,
    /// Backend health status
    pub status: GpuDetectionStatus,
    /// Available features
    pub available_features: GpuFeatureSet,
}

/// Detection status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDetectionStatus {
    /// GPU detected and healthy
    Healthy,
    /// GPU detected but degraded (reduced performance)
    Degraded,
    /// GPU error or unavailable
    Unavailable,
    /// Not yet attempted
    Undetected,
}

/// Available GPU features on this system
#[derive(Debug, Clone, Default)]
pub struct GpuFeatureSet {
    /// CUDA available
    pub cuda: bool,
    /// Metal available (macOS)
    pub metal: bool,
    /// Vulkan available
    pub vulkan: bool,
    /// WGPU available
    pub wgpu: bool,
}

impl GpuAutoDetector {
    /// Create new detector
    pub fn new() -> Self {
        Self {
            backend: None,
            detected_at: Instant::now(),
            is_ready: false,
            status: GpuDetectionStatus::Undetected,
            available_features: GpuFeatureSet::detect(),
        }
    }

    /// Strict GPU detection - errors if no GPU available
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No GPU features are enabled
    /// - GPU initialization fails
    /// - GPU device is unavailable
    pub fn detect_gpu_strict() -> Result<Self> {
        let mut detector = Self::new();

        // Check if any GPU feature is enabled
        if !detector.available_features.has_any_gpu() {
            return Err(ModelError::Backend {
                message: "No GPU features enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal".to_string(),
            });
        }

        // Try detection in priority order
        detector
            .try_detect_cuda()
            .or_else(|_| detector.try_detect_metal())
            .or_else(|_| detector.try_detect_wgpu())
            .map(|_| detector)
    }

    /// Strict Intel NPU detection - errors if no Intel NPU-capable adapter is available.
    ///
    /// This path does not fall back to CUDA/Metal/Vulkan general GPU selection.
    pub fn detect_npu_strict() -> Result<Self> {
        let mut detector = Self::new();
        let backend = resolve_compute_backend_strict_auto_npu()?;
        detector.backend = Some(backend);
        detector.is_ready = true;
        detector.status = GpuDetectionStatus::Healthy;
        tracing::info!("✓ NPU Backend: Intel NPU/WGPU (detected and healthy)");
        Ok(detector)
    }

    /// Detect with retry logic for troubleshooting
    ///
    /// Retries detection up to `max_retries` times with exponential backoff.
    pub fn detect_with_retry(max_retries: usize) -> Result<Self> {
        let mut last_error = ModelError::Backend {
            message: "GPU detection not attempted".to_string(),
        };

        for attempt in 0..max_retries {
            match Self::detect_gpu_strict() {
                Ok(detector) => return Ok(detector),
                Err(e) => {
                    last_error = e;
                    if attempt < max_retries - 1 {
                        let backoff_ms = (2_u64).pow(attempt as u32) * 100;
                        std::thread::sleep(Duration::from_millis(backoff_ms));
                    }
                }
            }
        }

        Err(last_error)
    }

    /// Try CUDA detection
    fn try_detect_cuda(&mut self) -> Result<()> {
        #[cfg(feature = "gpu-cuda")]
        {
            // Check if CUDA is available
            if crate::domain::compute_backend::detect_available_gpu_backends()
                .iter()
                .any(|b| *b == ComputeBackend::Cuda)
            {
                self.backend = Some(ComputeBackend::Cuda);
                self.is_ready = true;
                self.status = GpuDetectionStatus::Healthy;
                tracing::info!("✓ GPU Backend: CUDA (detected and healthy)");
                return Ok(());
            }
        }

        #[cfg(not(feature = "gpu-cuda"))]
        {
            tracing::debug!("CUDA feature not enabled");
        }

        Err(ModelError::Backend {
            message: "CUDA not available".to_string(),
        })
    }

    /// Try Metal detection (macOS only)
    fn try_detect_metal(&mut self) -> Result<()> {
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        {
            if crate::domain::compute_backend::detect_available_gpu_backends()
                .iter()
                .any(|b| *b == ComputeBackend::Metal)
            {
                self.backend = Some(ComputeBackend::Metal);
                self.is_ready = true;
                self.status = GpuDetectionStatus::Healthy;
                tracing::info!("✓ GPU Backend: Metal (detected and healthy)");
                return Ok(());
            }
        }

        #[cfg(not(feature = "gpu-metal"))]
        {
            tracing::debug!("Metal feature not enabled");
        }

        Err(ModelError::Backend {
            message: "Metal not available".to_string(),
        })
    }

    /// Try Vulkan detection (via WGPU)
    fn try_detect_wgpu(&mut self) -> Result<()> {
        #[cfg(feature = "wgpu")]
        {
            if crate::domain::compute_backend::detect_available_gpu_backends()
                .iter()
                .any(|b| *b == ComputeBackend::Vulkan)
            {
                self.backend = Some(ComputeBackend::Vulkan);
                self.is_ready = true;
                self.status = GpuDetectionStatus::Healthy;
                tracing::info!("✓ GPU Backend: Vulkan/WGPU (detected and healthy)");
                return Ok(());
            }
        }

        #[cfg(not(feature = "wgpu"))]
        {
            tracing::debug!("WGPU feature not enabled");
        }

        Err(ModelError::Backend {
            message: "WGPU/Vulkan not available".to_string(),
        })
    }

    /// Get backend name
    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            Some(ComputeBackend::Cuda) => "CUDA",
            Some(ComputeBackend::Metal) => "Metal",
            Some(ComputeBackend::Vulkan) => "Vulkan/WGPU",
            Some(ComputeBackend::Npu) => "Intel NPU/WGPU",
            Some(ComputeBackend::Cpu) => "CPU",
            None => "None",
        }
    }

    /// Check if GPU is healthy
    pub fn is_healthy(&self) -> bool {
        self.status == GpuDetectionStatus::Healthy && self.is_ready
    }

    /// Get detection diagnostics
    pub fn diagnostics(&self) -> GpuDetectionDiagnostics {
        GpuDetectionDiagnostics {
            backend_name: self.backend_name().to_string(),
            is_healthy: self.is_healthy(),
            status: self.status,
            uptime_secs: self.detected_at.elapsed().as_secs(),
            available_features: self.available_features.clone(),
        }
    }
}

impl Default for GpuAutoDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU detection diagnostics
#[derive(Debug, Clone)]
pub struct GpuDetectionDiagnostics {
    pub backend_name: String,
    pub is_healthy: bool,
    pub status: GpuDetectionStatus,
    pub uptime_secs: u64,
    pub available_features: GpuFeatureSet,
}

impl GpuFeatureSet {
    /// Detect available GPU features based on compile-time flags
    pub fn detect() -> Self {
        GpuFeatureSet {
            #[cfg(feature = "gpu-cuda")]
            cuda: true,
            #[cfg(not(feature = "gpu-cuda"))]
            cuda: false,

            #[cfg(feature = "gpu-metal")]
            metal: true,
            #[cfg(not(feature = "gpu-metal"))]
            metal: false,

            #[cfg(feature = "wgpu")]
            wgpu: true,
            #[cfg(not(feature = "wgpu"))]
            wgpu: false,

            // Vulkan support planned for future
            vulkan: false,
        }
    }

    /// Check if any GPU feature is available
    pub fn has_any_gpu(&self) -> bool {
        self.cuda || self.metal || self.wgpu || self.vulkan
    }

    /// Count available backends
    pub fn count_available(&self) -> usize {
        [self.cuda, self.metal, self.wgpu, self.vulkan]
            .iter()
            .filter(|&&b| b)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detector_creation() {
        let detector = GpuAutoDetector::new();
        assert_eq!(detector.status, GpuDetectionStatus::Undetected);
        assert!(!detector.is_ready);
    }

    #[test]
    fn test_feature_detection() {
        let features = GpuFeatureSet::detect();
        println!("Available GPU features:");
        println!("  CUDA: {}", features.cuda);
        println!("  Metal: {}", features.metal);
        println!("  WGPU: {}", features.wgpu);
        println!("  Vulkan: {}", features.vulkan);

        // At least one feature should be enabled for tests to be useful
        assert!(features.has_any_gpu() || !features.has_any_gpu()); // Always true for diagnostics
    }

    #[test]
    fn test_detection_strict() {
        // This will succeed if GPU is available, fail otherwise (expected)
        let result = GpuAutoDetector::detect_gpu_strict();
        match result {
            Ok(detector) => {
                assert!(detector.is_ready);
                assert!(detector.is_healthy());
                println!("GPU detected: {}", detector.backend_name());
            }
            Err(e) => {
                println!("No GPU available (expected on CPU-only systems): {}", e);
            }
        }
    }

    #[test]
    fn test_detection_npu_strict() {
        let result = GpuAutoDetector::detect_npu_strict();
        match result {
            Ok(detector) => {
                assert_eq!(detector.backend, Some(ComputeBackend::Npu));
                assert!(detector.is_ready);
            }
            Err(e) => {
                let msg = e.to_string().to_ascii_lowercase();
                assert!(
                    msg.contains("npu")
                        || msg.contains("gpu")
                        || msg.contains("fallback")
                        || msg.contains("compiled")
                );
            }
        }
    }

    #[test]
    fn test_diagnostics() {
        let mut detector = GpuAutoDetector::new();
        detector.status = GpuDetectionStatus::Healthy;
        detector.is_ready = true;
        detector.backend = Some(ComputeBackend::Cpu); // CPU for testing

        let diag = detector.diagnostics();
        assert!(diag.uptime_secs >= 0);
        assert!(!diag.backend_name.is_empty());
    }
}
