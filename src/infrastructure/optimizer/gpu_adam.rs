//! GPU-Accelerated Adam Optimizer
//!
//! This module provides a GPU implementation of the Adam optimizer with AMSGrad
//! and AdamW variants. All optimizer state (moment estimates, parameters) remains
//! on the GPU, eliminating CPU-GPU data transfer during training.
//!
//! ## Mathematical Formulation
//!
//! ### Adam Algorithm
//!
//! The Adam optimizer updates parameters θ using first and second moment estimates:
//!
//! ```text
//! m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
//! v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
//! ```
//!
//! Bias correction:
//! ```text
//! m̂_t = m_t / (1 - β₁^t)
//! v̂_t = v_t / (1 - β₂^t)
//! ```
//!
//! Parameter update:
//! ```text
//! θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
//! ```
//!
//! ### AMSGrad Variant
//!
//! Maintains the maximum of all past squared gradients:
//! ```text
//! v̂_{max,t} = max(v̂_{max,t-1}, v̂_t)
//! θ_t = θ_{t-1} - η · m̂_t / (√v̂_{max,t} + ε)
//! ```
//!
//! ### AdamW (Decoupled Weight Decay)
//!
//! Decouples weight decay from gradient update:
//! ```text
//! θ_t = θ_{t-1} · (1 - λη) - η · m̂_t / (√v̂_t + ε)
//! ```
//!
//! ## Performance Benefits
//!
//! - **Zero CPU-GPU Transfer**: Parameters and optimizer state remain on GPU
//! - **Parallel Updates**: All elements updated simultaneously via compute shaders
//! - **Memory Efficiency**: Reuses GPU buffers across training steps
//!
//! ## Example
//!
//! ```rust,ignore
//! use crate::infrastructure::optimizer::gpu_adam::GpuAdam;
//! use crate::domain::compute::GpuDevice;
//!
//! // Create GPU Adam optimizer
//! let mut optimizer = GpuAdam::new(device, param_count, 0.9, 0.999, 1e-8)?;
//!
//! // Enable AMSGrad
//! optimizer.set_amsgrad(true);
//!
//! // Perform optimization step (all on GPU)
//! optimizer.step(&mut params_gpu, &grads_gpu, 0.001)?;
//! ```

use crate::common::errors::{ModelError, Result};
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, GpuDevice};

/// Configuration for GPU Adam optimizer hyperparameters
#[derive(Debug, Clone, Copy)]
pub struct GpuAdamConfig {
    /// Exponential decay rate for first moment (default: 0.9)
    pub beta1: f32,
    /// Exponential decay rate for second moment (default: 0.999)
    pub beta2: f32,
    /// Numerical stability constant (default: 1e-8)
    pub epsilon: f32,
    /// Weight decay coefficient for AdamW (default: 0.0)
    pub weight_decay: f32,
    /// Use decoupled weight decay (AdamW style)
    pub use_decoupled_wd: bool,
    /// Enable AMSGrad variant
    pub use_amsgrad: bool,
}

impl Default for GpuAdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            use_decoupled_wd: false,
            use_amsgrad: false,
        }
    }
}

impl GpuAdamConfig {
    /// Create AdamW configuration with decoupled weight decay
    #[must_use]
    pub fn adamw(weight_decay: f32) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            use_decoupled_wd: true,
            use_amsgrad: false,
        }
    }

    /// Create AMSGrad configuration
    #[must_use]
    pub fn amsgrad() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            use_decoupled_wd: false,
            use_amsgrad: true,
        }
    }
}

/// GPU-resident Adam optimizer state
///
/// Maintains all optimizer state on the GPU for zero-copy training.
/// The first moment (m), second moment (v), and optionally v_max buffers
/// are allocated once and reused across all optimization steps.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuAdam {
    /// GPU device reference for buffer allocation and kernel dispatch
    device: Arc<Mutex<GpuDevice>>,
    /// Number of parameters being optimized
    param_count: usize,
    /// Optimizer hyperparameters
    config: GpuAdamConfig,
    /// Current timestep (for bias correction)
    timestep: u32,
    /// First moment estimate (m) - GPU buffer
    m_buffer: GpuBuffer,
    /// Second moment estimate (v) - GPU buffer
    v_buffer: GpuBuffer,
    /// Maximum second moment for AMSGrad (v_max) - GPU buffer (optional)
    v_max_buffer: Option<GpuBuffer>,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuAdam {
    /// Allocate a GPU buffer and initialize it to zeros.
    ///
    /// Prefers GPU-side fill to avoid host allocation and upload. Falls back to a
    /// CPU zero upload for backends that do not yet implement `fill_f32`.
    fn allocate_zeroed_buffer(gpu: &mut GpuDevice, param_count: usize) -> Result<GpuBuffer> {
        let size_bytes = param_count * std::mem::size_of::<f32>();
        let mut buffer = gpu.allocate(size_bytes)?;
        Self::zero_buffer(gpu, &mut buffer, param_count)?;
        Ok(buffer)
    }

    /// Reset an existing GPU buffer to all zeros.
    fn zero_buffer(gpu: &mut GpuDevice, buffer: &mut GpuBuffer, param_count: usize) -> Result<()> {
        match gpu.fill_f32(buffer, 0.0) {
            Ok(()) => Ok(()),
            Err(_) => {
                // Compatibility path for backends without a native fill kernel yet.
                let zeros = vec![0.0f32; param_count];
                gpu.upload(&zeros, buffer)
            }
        }
    }

    /// Create a new GPU Adam optimizer with default hyperparameters
    ///
    /// # Arguments
    /// * `device` - GPU device for buffer allocation
    /// * `param_count` - Number of parameters to optimize
    ///
    /// # Errors
    /// Returns error if GPU buffer allocation fails
    pub fn new(device: Arc<Mutex<GpuDevice>>, param_count: usize) -> Result<Self> {
        Self::with_config(device, param_count, GpuAdamConfig::default())
    }

    /// Create a new GPU Adam optimizer with custom hyperparameters
    ///
    /// # Arguments
    /// * `device` - GPU device for buffer allocation
    /// * `param_count` - Number of parameters to optimize
    /// * `config` - Optimizer hyperparameters
    ///
    /// # Errors
    /// Returns error if GPU buffer allocation fails
    pub fn with_config(
        device: Arc<Mutex<GpuDevice>>,
        param_count: usize,
        config: GpuAdamConfig,
    ) -> Result<Self> {
        let mut gpu = device
            .lock()
            .map_err(|_| ModelError::Backend {
                message: "Failed to lock GPU device for GpuAdam initialization".to_string(),
            })?;

        let m_buffer = Self::allocate_zeroed_buffer(&mut gpu, param_count)?;
        let v_buffer = Self::allocate_zeroed_buffer(&mut gpu, param_count)?;

        // Allocate v_max buffer for AMSGrad if needed
        let v_max_buffer = if config.use_amsgrad {
            Some(Self::allocate_zeroed_buffer(&mut gpu, param_count)?)
        } else {
            None
        };

        drop(gpu);

        Ok(Self {
            device,
            param_count,
            config,
            timestep: 0,
            m_buffer,
            v_buffer,
            v_max_buffer,
        })
    }

    /// Create a GPU Adam optimizer with AMSGrad enabled
    pub fn new_amsgrad(device: Arc<Mutex<GpuDevice>>, param_count: usize) -> Result<Self> {
        Self::with_config(device, param_count, GpuAdamConfig::amsgrad())
    }

    /// Create a GPU AdamW optimizer with decoupled weight decay
    pub fn new_adamw(
        device: Arc<Mutex<GpuDevice>>,
        param_count: usize,
        weight_decay: f32,
    ) -> Result<Self> {
        Self::with_config(device, param_count, GpuAdamConfig::adamw(weight_decay))
    }

    /// Enable or disable AMSGrad variant
    ///
    /// Allocates or deallocates the v_max buffer as needed.
    pub fn set_amsgrad(&mut self, enable: bool) -> Result<()> {
        if enable && self.v_max_buffer.is_none() {
            let mut gpu = self.device.lock().map_err(|_| ModelError::Backend {
                message: "Failed to lock GPU device for AMSGrad buffer allocation".to_string(),
            })?;
            self.v_max_buffer = Some(Self::allocate_zeroed_buffer(&mut gpu, self.param_count)?);
        } else if !enable {
            if let Some(buf) = self.v_max_buffer.take() {
                let mut gpu = self.device.lock().map_err(|_| ModelError::Backend {
                    message: "Failed to lock GPU device for AMSGrad buffer deallocation"
                        .to_string(),
                })?;
                gpu.deallocate(buf);
            }
        }
        self.config.use_amsgrad = enable;
        Ok(())
    }

    /// Set weight decay parameters
    pub fn set_weight_decay(&mut self, weight_decay: f32, decoupled: bool) {
        self.config.weight_decay = weight_decay;
        self.config.use_decoupled_wd = decoupled;
    }

    /// Reset optimizer state (useful for restarting training)
    ///
    /// Resets timestep to 0 and zeros all moment buffers.
    pub fn reset(&mut self) -> Result<()> {
        self.timestep = 0;

        let mut gpu = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock GPU device for GpuAdam reset".to_string(),
        })?;

        Self::zero_buffer(&mut gpu, &mut self.m_buffer, self.param_count)?;
        Self::zero_buffer(&mut gpu, &mut self.v_buffer, self.param_count)?;

        if let Some(ref mut v_max) = self.v_max_buffer {
            Self::zero_buffer(&mut gpu, v_max, self.param_count)?;
        }

        Ok(())
    }

    /// Get the number of parameters
    #[inline]
    pub fn param_count(&self) -> usize {
        self.param_count
    }

    /// Get current timestep
    #[inline]
    pub fn timestep(&self) -> u32 {
        self.timestep
    }

    /// Check if AMSGrad is enabled
    #[inline]
    pub fn is_amsgrad(&self) -> bool {
        self.config.use_amsgrad
    }

    /// Check if decoupled weight decay (AdamW) is enabled
    #[inline]
    pub fn is_decoupled_wd(&self) -> bool {
        self.config.use_decoupled_wd && self.config.weight_decay > 0.0
    }

    /// Get weight decay coefficient
    #[inline]
    pub fn weight_decay(&self) -> f32 {
        self.config.weight_decay
    }

    /// Perform optimization step on GPU
    ///
    /// Updates parameters using the Adam algorithm. All computation happens on GPU.
    ///
    /// # Arguments
    /// * `params` - GPU buffer containing parameters (modified in-place)
    /// * `grads` - GPU buffer containing gradients
    /// * `lr` - Learning rate
    ///
    /// # Errors
    /// Returns error if GPU kernel dispatch fails
    pub fn step(
        &mut self,
        params: &mut GpuBuffer,
        grads: &GpuBuffer,
        lr: f32,
    ) -> Result<()> {
        // Early exit for zero learning rate
        if lr == 0.0 {
            return Ok(());
        }

        self.timestep += 1;

        let mut gpu = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock GPU device for Adam step".to_string(),
        })?;

        // Compute bias correction factors
        let inv_bias1 = 1.0 / (1.0 - self.config.beta1.powi(self.timestep as i32));
        let inv_bias2 = 1.0 / (1.0 - self.config.beta2.powi(self.timestep as i32));

        // Dispatch Adam kernel
        gpu.adam_step(
            params,
            grads,
            &mut self.m_buffer,
            &mut self.v_buffer,
            self.v_max_buffer.as_mut(),
            lr,
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon,
            inv_bias1,
            inv_bias2,
            self.config.weight_decay,
            self.config.use_decoupled_wd,
            self.config.use_amsgrad,
            self.param_count,
        )
    }

    /// Get first moment buffer (for debugging/inspection)
    pub fn m_buffer(&self) -> &GpuBuffer {
        &self.m_buffer
    }

    /// Get second moment buffer (for debugging/inspection)
    pub fn v_buffer(&self) -> &GpuBuffer {
        &self.v_buffer
    }

    /// Get v_max buffer for AMSGrad (for debugging/inspection)
    pub fn v_max_buffer(&self) -> Option<&GpuBuffer> {
        self.v_max_buffer.as_ref()
    }

    /// Download first moment to CPU (for checkpointing)
    pub fn download_m(&self) -> Result<Vec<f32>> {
        let mut gpu = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock GPU device for m download".to_string(),
        })?;
        let mut data = vec![0.0f32; self.param_count];
        gpu.download(&self.m_buffer, &mut data)?;
        Ok(data)
    }

    /// Download second moment to CPU (for checkpointing)
    pub fn download_v(&self) -> Result<Vec<f32>> {
        let mut gpu = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock GPU device for v download".to_string(),
        })?;
        let mut data = vec![0.0f32; self.param_count];
        gpu.download(&self.v_buffer, &mut data)?;
        Ok(data)
    }

    /// Upload first moment from CPU (for resuming from checkpoint)
    pub fn upload_m(&mut self, data: &[f32]) -> Result<()> {
        if data.len() != self.param_count {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "m buffer size mismatch: expected {}, got {}",
                    self.param_count,
                    data.len()
                ),
            });
        }
        let mut gpu = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock GPU device for m upload".to_string(),
        })?;
        gpu.upload(data, &mut self.m_buffer)
    }

    /// Upload second moment from CPU (for resuming from checkpoint)
    pub fn upload_v(&mut self, data: &[f32]) -> Result<()> {
        if data.len() != self.param_count {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "v buffer size mismatch: expected {}, got {}",
                    self.param_count,
                    data.len()
                ),
            });
        }
        let mut gpu = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock GPU device for v upload".to_string(),
        })?;
        gpu.upload(data, &mut self.v_buffer)
    }
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl Drop for GpuAdam {
    fn drop(&mut self) {
        // Deallocate GPU buffers
        if let Ok(mut gpu) = self.device.lock() {
            gpu.deallocate(self.m_buffer);
            gpu.deallocate(self.v_buffer);
            if let Some(v_max) = self.v_max_buffer.take() {
                gpu.deallocate(v_max);
            }
        }
    }
}

// ============================================================================
// CPU Fallback (for non-GPU builds)
// ============================================================================

/// CPU-only fallback for non-GPU builds
#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
pub struct GpuAdam {
    _param_count: usize,
    _config: GpuAdamConfig,
    _timestep: u32,
}

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
impl GpuAdam {
    /// Create a new GPU Adam optimizer (CPU fallback - returns error)
    pub fn new(_device: Arc<Mutex<()>>, _param_count: usize) -> Result<Self> {
        Err(ModelError::Backend {
            message: "GpuAdam requires GPU features. Rebuild with --features gpu-wgpu, gpu-cuda, or gpu-metal".to_string(),
        })
    }

    /// Create with config (CPU fallback - returns error)
    pub fn with_config(
        _device: Arc<Mutex<()>>,
        _param_count: usize,
        _config: GpuAdamConfig,
    ) -> Result<Self> {
        Err(ModelError::Backend {
            message: "GpuAdam requires GPU features. Rebuild with --features gpu-wgpu, gpu-cuda, or gpu-metal".to_string(),
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
    fn test_adam_config_default() {
        let config = GpuAdamConfig::default();
        assert!((config.beta1 - 0.9).abs() < 1e-6);
        assert!((config.beta2 - 0.999).abs() < 1e-6);
        assert!((config.epsilon - 1e-8).abs() < 1e-12);
        assert!(!config.use_amsgrad);
        assert!(!config.use_decoupled_wd);
        assert!((config.weight_decay - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_adam_config_adamw() {
        let config = GpuAdamConfig::adamw(0.01);
        assert!(config.use_decoupled_wd);
        assert!((config.weight_decay - 0.01).abs() < 1e-6);
        assert!(!config.use_amsgrad);
    }

    #[test]
    fn test_adam_config_amsgrad() {
        let config = GpuAdamConfig::amsgrad();
        assert!(config.use_amsgrad);
        assert!(!config.use_decoupled_wd);
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_gpu_adam_creation() {
        use crate::domain::compute::GpuDevice;
        use crate::domain::compute_backend::ComputeBackend;

        // Try to create GPU device
        let device = GpuDevice::new(ComputeBackend::Vulkan);
        if device.is_err() {
            // Skip test if GPU not available
            eprintln!("Skipping test: GPU device not available");
            return;
        }

        let device = Arc::new(Mutex::new(device.unwrap()));
        let param_count = 100;

        let optimizer = GpuAdam::new(device, param_count);
        assert!(optimizer.is_ok());

        let opt = optimizer.unwrap();
        assert_eq!(opt.param_count(), param_count);
        assert_eq!(opt.timestep(), 0);
        assert!(!opt.is_amsgrad());
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_gpu_adam_amsgrad() {
        use crate::domain::compute::GpuDevice;
        use crate::domain::compute_backend::ComputeBackend;

        let device = GpuDevice::new(ComputeBackend::Vulkan);
        if device.is_err() {
            eprintln!("Skipping test: GPU device not available");
            return;
        }

        let device = Arc::new(Mutex::new(device.unwrap()));
        let param_count = 100;

        let optimizer = GpuAdam::new_amsgrad(device, param_count);
        assert!(optimizer.is_ok());

        let opt = optimizer.unwrap();
        assert!(opt.is_amsgrad());
        assert!(opt.v_max_buffer().is_some());
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_gpu_adam_reset() {
        use crate::domain::compute::GpuDevice;
        use crate::domain::compute_backend::ComputeBackend;

        let device = GpuDevice::new(ComputeBackend::Vulkan);
        if device.is_err() {
            eprintln!("Skipping test: GPU device not available");
            return;
        }

        let device = Arc::new(Mutex::new(device.unwrap()));
        let param_count = 100;

        let mut optimizer = GpuAdam::new(device, param_count).unwrap();
        optimizer.timestep = 10;

        let result = optimizer.reset();
        assert!(result.is_ok());
        assert_eq!(optimizer.timestep(), 0);
    }
}
