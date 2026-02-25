//! GPU Training Pipeline
//!
//! This module provides end-to-end GPU training with zero CPU-GPU data transfer.
//! All operations (forward pass, backward pass, optimizer step) remain on GPU.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        GpuTrainingPipeline                          │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
//! │  │ Input Buffer │  │ Param Buffer │  │ Optimizer State (GpuAdam)│  │
//! │  │    (GPU)     │  │    (GPU)     │  │   m, v, v_max (GPU)      │  │
//! │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
//! │                                                                      │
//! │  Training Loop (all GPU):                                           │
//! │  1. Forward pass → output_buffer (GPU)                              │
//! │  2. Loss computation → loss_buffer (GPU)                            │
//! │  3. Backward pass → grad_buffer (GPU)                               │
//! │  4. Optimizer step → updated params (GPU)                           │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Performance Benefits
//!
//! - **Zero CPU-GPU Transfer**: All tensors remain on GPU throughout training
//! - **Fused Operations**: Combined kernels where possible
//! - **Memory Efficiency**: Buffer reuse across training steps
//! - **Parallel Optimization**: All parameters updated simultaneously

use crate::common::errors::{ModelError, Result};
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::{
    domain::compute::{GpuBuffer, GpuDevice},
    infrastructure::optimizer::{GpuAdam, GpuAdamConfig},
};

/// Configuration for GPU training pipeline
#[derive(Debug, Clone)]
pub struct GpuTrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Number of epochs
    pub epochs: usize,
    /// Micro-batch size
    pub batch_size: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Warmup epochs for learning rate
    pub warmup_epochs: usize,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Use AdamW (decoupled weight decay)
    pub use_adamw: bool,
    /// Use AMSGrad variant
    pub use_amsgrad: bool,
    /// Apply sign-preserving log scaling to gradients before optimizer step
    pub use_log_gradient_scaling: bool,
    /// Strength parameter for log scaling: sign(g) * log1p(alpha*|g|)/alpha
    pub log_gradient_alpha: f32,
    /// Enable EMA-smoothed bidirectional adaptive LARS trust-ratio learning-rate scaling
    pub use_bidirectional_adaptive_lars: bool,
    /// Numerical stabilizer for adaptive LARS trust ratio
    pub adaptive_lars_epsilon: f32,
    /// Minimum trust ratio (allows down-scaling without clipping)
    pub adaptive_lars_trust_min: f32,
    /// Maximum trust ratio (allows up-scaling while bounded)
    pub adaptive_lars_trust_max: f32,
    /// EMA decay for param/grad norms used by adaptive LARS
    pub adaptive_lars_ema_decay: f32,
    /// Learning rate minimum (for cosine annealing)
    pub lr_min_ratio: f32,
}

impl Default for GpuTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 10,
            batch_size: 32,
            gradient_accumulation_steps: 1,
            warmup_epochs: 0,
            weight_decay: 0.0,
            use_adamw: false,
            use_amsgrad: false,
            use_log_gradient_scaling: true,
            log_gradient_alpha: 1.0,
            use_bidirectional_adaptive_lars: true,
            adaptive_lars_epsilon: 1e-8,
            adaptive_lars_trust_min: 0.25,
            adaptive_lars_trust_max: 4.0,
            adaptive_lars_ema_decay: 0.95,
            lr_min_ratio: 0.1,
        }
    }
}

impl GpuTrainingConfig {
    /// Create config for AdamW training
    pub fn adamw(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            weight_decay,
            use_adamw: true,
            ..Default::default()
        }
    }

    /// Create config for AMSGrad training
    pub fn amsgrad(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            use_amsgrad: true,
            ..Default::default()
        }
    }

    /// Set number of epochs
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set gradient accumulation steps
    pub fn with_gradient_accumulation(mut self, steps: usize) -> Self {
        self.gradient_accumulation_steps = steps;
        self
    }

    /// Set warmup epochs
    pub fn with_warmup(mut self, epochs: usize) -> Self {
        self.warmup_epochs = epochs;
        self
    }

    /// Enable/disable sign-preserving log gradient scaling.
    pub fn with_log_gradient_scaling(mut self, enabled: bool, alpha: f32) -> Self {
        self.use_log_gradient_scaling = enabled;
        self.log_gradient_alpha = alpha.max(1e-8);
        self
    }

    /// Configure bidirectional adaptive LARS trust-ratio scaling.
    pub fn with_bidirectional_adaptive_lars(
        mut self,
        enabled: bool,
        trust_min: f32,
        trust_max: f32,
        ema_decay: f32,
    ) -> Self {
        self.use_bidirectional_adaptive_lars = enabled;
        self.adaptive_lars_trust_min = trust_min.max(0.0);
        self.adaptive_lars_trust_max = trust_max.max(self.adaptive_lars_trust_min);
        self.adaptive_lars_ema_decay = ema_decay.clamp(0.0, 0.9999);
        self
    }
}

/// Learning rate scheduler for GPU training
#[derive(Debug, Clone, Copy)]
pub enum LrScheduler {
    /// Constant learning rate
    Constant,
    /// Linear warmup then cosine annealing
    WarmupCosine {
        warmup_epochs: usize,
        total_epochs: usize,
        lr_min_ratio: f32,
    },
    /// Linear warmup then linear decay
    WarmupLinear {
        warmup_epochs: usize,
        total_epochs: usize,
        lr_min_ratio: f32,
    },
    /// Exponential decay
    ExponentialDecay { gamma: f32 },
}

impl LrScheduler {
    /// Compute learning rate for given epoch
    pub fn get_lr(&self, epoch: usize, base_lr: f32) -> f32 {
        match self {
            Self::Constant => base_lr,

            Self::WarmupCosine {
                warmup_epochs,
                total_epochs,
                lr_min_ratio,
            } => {
                if epoch < *warmup_epochs {
                    // Linear warmup
                    base_lr * ((epoch + 1) as f32 / (*warmup_epochs) as f32)
                } else {
                    // Cosine annealing
                    let t = (epoch - warmup_epochs) as f32;
                    let t_max = (total_epochs - warmup_epochs).max(1) as f32;
                    let lr_min = base_lr * lr_min_ratio;
                    lr_min + 0.5 * (base_lr - lr_min) * (1.0 + (std::f32::consts::PI * t / t_max).cos())
                }
            }

            Self::WarmupLinear {
                warmup_epochs,
                total_epochs,
                lr_min_ratio,
            } => {
                if epoch < *warmup_epochs {
                    // Linear warmup
                    base_lr * ((epoch + 1) as f32 / (*warmup_epochs) as f32)
                } else {
                    // Linear decay
                    let t = (epoch - warmup_epochs) as f32;
                    let t_max = (total_epochs - warmup_epochs).max(1) as f32;
                    let lr_min = base_lr * lr_min_ratio;
                    lr_min + (base_lr - lr_min) * (1.0 - t / t_max)
                }
            }

            Self::ExponentialDecay { gamma } => base_lr * gamma.powi(epoch as i32),
        }
    }

    /// Create warmup + cosine annealing scheduler
    pub fn warmup_cosine(warmup_epochs: usize, total_epochs: usize) -> Self {
        Self::WarmupCosine {
            warmup_epochs,
            total_epochs,
            lr_min_ratio: 0.1,
        }
    }

    /// Create warmup + linear decay scheduler
    pub fn warmup_linear(warmup_epochs: usize, total_epochs: usize) -> Self {
        Self::WarmupLinear {
            warmup_epochs,
            total_epochs,
            lr_min_ratio: 0.0,
        }
    }
}

/// GPU-resident training state for a single parameter group
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuParameterGroup {
    /// Parameter buffer on GPU
    pub params: GpuBuffer,
    /// Gradient buffer on GPU
    pub grads: GpuBuffer,
    /// Gradient accumulation buffer (for gradient accumulation)
    pub grad_accumulator: GpuBuffer,
    /// Number of parameters in this group
    pub param_count: usize,
    /// Accumulation counter
    pub accum_count: usize,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuParameterGroup {
    #[inline]
    fn zero_buffer(device: &mut GpuDevice, buffer: &mut GpuBuffer, param_count: usize) -> Result<()> {
        match device.fill_f32(buffer, 0.0) {
            Ok(()) => Ok(()),
            Err(_) => {
                let zeros = vec![0.0f32; param_count];
                device.upload(&zeros, buffer)
            }
        }
    }

    /// Create a new GPU parameter group
    pub fn new(device: &mut GpuDevice, param_count: usize) -> Result<Self> {
        let size_bytes = param_count * std::mem::size_of::<f32>();

        let params = device.allocate(size_bytes)?;
        let mut grads = device.allocate(size_bytes)?;
        let mut grad_accumulator = device.allocate(size_bytes)?;

        // Initialize gradients/accumulator to zero (GPU-first)
        Self::zero_buffer(device, &mut grads, param_count)?;
        Self::zero_buffer(device, &mut grad_accumulator, param_count)?;

        Ok(Self {
            params,
            grads,
            grad_accumulator,
            param_count,
            accum_count: 0,
        })
    }

    /// Accumulate gradients (add current grads to accumulator) on GPU.
    pub fn accumulate_grads(&mut self, device: &mut GpuDevice) -> Result<()> {
        device.add_scaled(
            1.0,
            &self.grads,
            &mut self.grad_accumulator,
            self.param_count,
        )?;
        self.accum_count += 1;
        Ok(())
    }

    /// Scale accumulated gradients by accumulation count and copy to grads
    pub fn finalize_accumulation(&mut self, device: &mut GpuDevice) -> Result<()> {
        // Copy accumulated gradients into the active grads buffer
        device.copy_within_device(&self.grad_accumulator, &mut self.grads, self.param_count)?;

        // Average if multiple accumulation steps were used
        if self.accum_count > 1 {
            let scale = 1.0 / self.accum_count as f32;
            device.scale(scale, &mut self.grads, self.param_count)?;
        }
        Ok(())
    }

    /// Reset gradient accumulator
    pub fn reset_accumulator(&mut self, device: &mut GpuDevice) -> Result<()> {
        Self::zero_buffer(device, &mut self.grad_accumulator, self.param_count)?;
        self.accum_count = 0;
        Ok(())
    }
}

/// GPU Training Pipeline
///
/// Manages end-to-end GPU training with zero CPU-GPU transfer.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuTrainingPipeline {
    /// GPU device
    device: Arc<Mutex<GpuDevice>>,
    /// Training configuration
    config: GpuTrainingConfig,
    /// Adam optimizer
    optimizer: GpuAdam,
    /// Scratch buffer for norm computations / temporary reductions
    norm_scratch: GpuBuffer,
    /// Work buffer for gradient transforms (e.g. log scaling) before optimizer step
    grad_work: GpuBuffer,
    /// Learning rate scheduler
    lr_scheduler: LrScheduler,
    /// Current epoch
    current_epoch: usize,
    /// Global step counter
    global_step: usize,
    /// EMA of parameter norm for adaptive LARS
    ema_param_norm: Option<f32>,
    /// EMA of gradient norm for adaptive LARS
    ema_grad_norm: Option<f32>,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuTrainingPipeline {
    #[inline]
    fn ema_update(prev: Option<f32>, value: f32, decay: f32) -> f32 {
        match prev {
            Some(p) => decay * p + (1.0 - decay) * value,
            None => value,
        }
    }

    fn l2_norm_gpu(
        device: &mut GpuDevice,
        input: &GpuBuffer,
        scratch: &mut GpuBuffer,
        size: usize,
    ) -> Result<f32> {
        device.mul(input, input, scratch, size)?;
        let sum_sq = device.sum(scratch, size)?;
        Ok(sum_sq.max(0.0).sqrt())
    }

    fn signed_log1p_scale_cpu_fallback(
        device: &mut GpuDevice,
        buffer: &mut GpuBuffer,
        size: usize,
        alpha: f32,
    ) -> Result<()> {
        let mut host = vec![0.0f32; size];
        device.download(buffer, &mut host)?;
        let a = alpha.max(1e-8);
        for g in &mut host {
            let mag = (1.0 + a * g.abs()).ln() / a;
            *g = g.signum() * mag;
        }
        device.upload(&host, buffer)
    }

    /// Create a new GPU training pipeline
    pub fn new(
        device: Arc<Mutex<GpuDevice>>,
        param_count: usize,
        config: GpuTrainingConfig,
    ) -> Result<Self> {
        // Create optimizer with appropriate config
        let optimizer_config = if config.use_adamw {
            GpuAdamConfig::adamw(config.weight_decay)
        } else if config.use_amsgrad {
            GpuAdamConfig::amsgrad()
        } else {
            GpuAdamConfig {
                weight_decay: config.weight_decay,
                ..Default::default()
            }
        };

        let optimizer = GpuAdam::with_config(device.clone(), param_count, optimizer_config)?;

        let (norm_scratch, grad_work) = {
            let mut gpu = device.lock().map_err(|_| ModelError::Backend {
                message: "Failed to lock GPU device for training scratch allocation".to_string(),
            })?;
            let size_bytes = param_count * std::mem::size_of::<f32>();
            (gpu.allocate(size_bytes)?, gpu.allocate(size_bytes)?)
        };

        // Create learning rate scheduler
        let lr_scheduler = LrScheduler::warmup_cosine(config.warmup_epochs, config.epochs);

        Ok(Self {
            device,
            config,
            optimizer,
            norm_scratch,
            grad_work,
            lr_scheduler,
            current_epoch: 0,
            global_step: 0,
            ema_param_norm: None,
            ema_grad_norm: None,
        })
    }

    /// Get current learning rate
    pub fn current_lr(&self) -> f32 {
        self.lr_scheduler.get_lr(self.current_epoch, self.config.learning_rate)
    }

    /// Perform optimizer step
    ///
    /// This is the core optimization step that updates parameters on GPU.
    pub fn step(&mut self, params: &mut GpuBuffer, grads: &GpuBuffer) -> Result<()> {
        let base_lr = self.current_lr();
        let mut effective_lr = base_lr;
        let mut grads_for_step = *grads;

        if self.config.use_log_gradient_scaling || self.config.use_bidirectional_adaptive_lars {
            let device_arc = self.device.clone();
            let mut gpu = device_arc.lock().map_err(|_| ModelError::Backend {
                message: "Failed to lock GPU device for gradient conditioning".to_string(),
            })?;

            if self.config.use_log_gradient_scaling {
                gpu.copy_within_device(grads, &mut self.grad_work, self.optimizer.param_count())?;
                let alpha = self.config.log_gradient_alpha.max(1e-8);
                if gpu
                    .signed_log1p_scale(&mut self.grad_work, alpha, self.optimizer.param_count())
                    .is_err()
                {
                    Self::signed_log1p_scale_cpu_fallback(
                        &mut gpu,
                        &mut self.grad_work,
                        self.optimizer.param_count(),
                        alpha,
                    )?;
                }
                grads_for_step = self.grad_work;
            }

            if self.config.use_bidirectional_adaptive_lars {
                let grad_ref = if self.config.use_log_gradient_scaling {
                    &self.grad_work
                } else {
                    grads
                };

                let param_norm = Self::l2_norm_gpu(
                    &mut gpu,
                    params,
                    &mut self.norm_scratch,
                    self.optimizer.param_count(),
                )?;
                let grad_norm = Self::l2_norm_gpu(
                    &mut gpu,
                    grad_ref,
                    &mut self.norm_scratch,
                    self.optimizer.param_count(),
                )?;

                let decay = self.config.adaptive_lars_ema_decay.clamp(0.0, 0.9999);
                let param_ema = Self::ema_update(self.ema_param_norm, param_norm, decay);
                let grad_ema = Self::ema_update(self.ema_grad_norm, grad_norm, decay);
                self.ema_param_norm = Some(param_ema);
                self.ema_grad_norm = Some(grad_ema);

                let eps = self.config.adaptive_lars_epsilon.max(1e-12);
                let mut trust = (param_ema + eps) / (grad_ema + eps);
                if !trust.is_finite() {
                    trust = 1.0;
                }
                trust = trust.clamp(
                    self.config.adaptive_lars_trust_min,
                    self.config.adaptive_lars_trust_max,
                );
                effective_lr *= trust;
            }
        }

        // Perform Adam step on GPU using conditioned gradients / adaptive trust ratio
        self.optimizer.step(params, &grads_for_step, effective_lr)?;

        self.global_step += 1;
        Ok(())
    }

    /// Advance to next epoch
    pub fn advance_epoch(&mut self) {
        self.current_epoch += 1;
    }

    /// Reset optimizer state (for restarting training)
    pub fn reset(&mut self) -> Result<()> {
        self.optimizer.reset()?;
        self.current_epoch = 0;
        self.global_step = 0;
        self.ema_param_norm = None;
        self.ema_grad_norm = None;
        Ok(())
    }

    /// Get current epoch
    pub fn epoch(&self) -> usize {
        self.current_epoch
    }

    /// Get global step
    pub fn step_count(&self) -> usize {
        self.global_step
    }

    /// Get optimizer for direct access
    pub fn optimizer(&self) -> &GpuAdam {
        &self.optimizer
    }

    /// Get mutable optimizer
    pub fn optimizer_mut(&mut self) -> &mut GpuAdam {
        &mut self.optimizer
    }

    /// Save training state for checkpointing
    pub fn save_state(&self) -> Result<GpuTrainingState> {
        Ok(GpuTrainingState {
            epoch: self.current_epoch,
            global_step: self.global_step,
            m: self.optimizer.download_m()?,
            v: self.optimizer.download_v()?,
            timestep: self.optimizer.timestep(),
        })
    }

    /// Load training state from checkpoint
    pub fn load_state(&mut self, state: &GpuTrainingState) -> Result<()> {
        self.optimizer.upload_m(&state.m)?;
        self.optimizer.upload_v(&state.v)?;
        self.current_epoch = state.epoch;
        self.global_step = state.global_step;
        Ok(())
    }
}

/// Serializable training state for checkpointing
#[derive(Debug, Clone)]
pub struct GpuTrainingState {
    /// Current epoch
    pub epoch: usize,
    /// Global step count
    pub global_step: usize,
    /// First moment estimates
    pub m: Vec<f32>,
    /// Second moment estimates
    pub v: Vec<f32>,
    /// Optimizer timestep
    pub timestep: u32,
}

// ============================================================================
// CPU Fallback (for non-GPU builds)
// ============================================================================

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
pub struct GpuTrainingPipeline {
    _config: GpuTrainingConfig,
}

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
impl GpuTrainingPipeline {
    pub fn new(
        _device: Arc<Mutex<()>>,
        _param_count: usize,
        config: GpuTrainingConfig,
    ) -> Result<Self> {
        Err(ModelError::Backend {
            message: "GpuTrainingPipeline requires GPU features. Rebuild with --features gpu-wgpu, gpu-cuda, or gpu-metal".to_string(),
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
    fn test_lr_scheduler_constant() {
        let scheduler = LrScheduler::Constant;
        assert!((scheduler.get_lr(0, 0.001) - 0.001).abs() < 1e-6);
        assert!((scheduler.get_lr(10, 0.001) - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_lr_scheduler_warmup_cosine() {
        let scheduler = LrScheduler::warmup_cosine(5, 20);
        let base_lr = 0.001;

        // During warmup: linear increase
        let lr_0 = scheduler.get_lr(0, base_lr);
        let lr_4 = scheduler.get_lr(4, base_lr);
        assert!(lr_0 < lr_4);
        assert!((lr_4 - base_lr).abs() < 1e-6); // End of warmup = base_lr

        // After warmup: cosine decay
        let lr_10 = scheduler.get_lr(10, base_lr);
        let lr_19 = scheduler.get_lr(19, base_lr);
        assert!(lr_10 > lr_19); // Should decay
    }

    #[test]
    fn test_lr_scheduler_warmup_linear() {
        let scheduler = LrScheduler::warmup_linear(3, 10);
        let base_lr = 0.001;

        // During warmup
        let lr_0 = scheduler.get_lr(0, base_lr);
        let lr_2 = scheduler.get_lr(2, base_lr);
        assert!(lr_0 < lr_2);

        // After warmup: linear decay to 0
        let lr_9 = scheduler.get_lr(9, base_lr);
        assert!(lr_9 < base_lr);
    }

    #[test]
    fn test_lr_scheduler_exponential() {
        let scheduler = LrScheduler::ExponentialDecay { gamma: 0.9 };
        let base_lr = 0.001;

        let lr_0 = scheduler.get_lr(0, base_lr);
        let lr_1 = scheduler.get_lr(1, base_lr);
        let lr_2 = scheduler.get_lr(2, base_lr);

        assert!((lr_0 - base_lr).abs() < 1e-6);
        assert!((lr_1 - base_lr * 0.9).abs() < 1e-6);
        assert!((lr_2 - base_lr * 0.81).abs() < 1e-6);
    }

    #[test]
    fn test_training_config_default() {
        let config = GpuTrainingConfig::default();
        assert!((config.learning_rate - 0.001).abs() < 1e-6);
        assert_eq!(config.epochs, 10);
        assert_eq!(config.batch_size, 32);
        assert!(!config.use_adamw);
        assert!(!config.use_amsgrad);
        assert!(config.use_log_gradient_scaling);
        assert!(config.use_bidirectional_adaptive_lars);
    }

    #[test]
    fn test_training_config_adamw() {
        let config = GpuTrainingConfig::adamw(0.001, 0.01);
        assert!(config.use_adamw);
        assert!((config.weight_decay - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_training_config_amsgrad() {
        let config = GpuTrainingConfig::amsgrad(0.001);
        assert!(config.use_amsgrad);
    }

    #[test]
    fn test_training_config_builder() {
        let config = GpuTrainingConfig::default()
            .with_epochs(100)
            .with_batch_size(64)
            .with_gradient_accumulation(4)
            .with_warmup(10)
            .with_log_gradient_scaling(true, 0.5)
            .with_bidirectional_adaptive_lars(true, 0.2, 3.0, 0.9);

        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.gradient_accumulation_steps, 4);
        assert_eq!(config.warmup_epochs, 10);
        assert!(config.use_log_gradient_scaling);
        assert!((config.log_gradient_alpha - 0.5).abs() < 1e-6);
        assert!(config.use_bidirectional_adaptive_lars);
        assert!((config.adaptive_lars_trust_min - 0.2).abs() < 1e-6);
        assert!((config.adaptive_lars_trust_max - 3.0).abs() < 1e-6);
        assert!((config.adaptive_lars_ema_decay - 0.9).abs() < 1e-6);
    }
}
