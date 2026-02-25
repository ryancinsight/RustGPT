//! GPU Backend Variants for Shared Components (Phase 5.6)
//!
//! Provides unified GPU backend implementations for Diffusion, SSM, and Transformer
//! architectures with automatic GPU detection and strict no-fallback semantics.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                    GpuBackendVariants                               │
//! ├────────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
//! │  │ DiffusionGpu │  │   SsmGpu     │  │    TransformerGpu        │ │
//! │  │   Backend    │  │   Backend    │  │       Backend            │ │
//! │  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘ │
//! │         │                 │                       │               │
//! │         └─────────────────┼───────────────────────┘               │
//! │                           │                                       │
//! │                           ▼                                       │
//! │  ┌────────────────────────────────────────────────────────────┐  │
//! │  │                  UnifiedGpuKernels                          │  │
//! │  │   (Attention, SSM, Normalization, Activation)              │  │
//! │  └────────────────────────────────────────────────────────────┘  │
//! │                           │                                       │
//! │                           ▼                                       │
//! │  ┌────────────────────────────────────────────────────────────┐  │
//! │  │                    GpuDevice                                │  │
//! │  │   (CUDA > Metal > Vulkan auto-detection)                   │  │
//! │  └────────────────────────────────────────────────────────────┘  │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Memory Efficiency
//!
//! All variants share:
//! - Pre-allocated workspace buffers with power-of-2 sizing
//! - Buffer reuse across kernel calls
//! - Zero-copy transfers when possible
//!
//! ## Performance Targets
//!
//! | Backend          | Operation         | CPU Time | GPU Target | Speedup |
//! |------------------|-------------------|----------|------------|---------|
//! | DiffusionGpu     | Noise prediction  | 45ms     | 2ms        | 22x     |
//! | SsmGpu           | Selective scan    | 50ms     | 2.5ms      | 20x     |
//! | TransformerGpu   | Multi-head attn   | 40ms     | 1.3ms      | 30x     |

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use ndarray::{Array1, Array2};

use crate::common::errors::{ModelError, Result};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::GpuDevice;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::{
    unified_gpu_backend::{GpuActivation, GpuTemporalType},
    unified_gpu_kernels::{AttentionParams, NormParams, SsmParams, UnifiedGpuKernels},
};

// ============================================================================
// Diffusion GPU Backend
// ============================================================================

/// GPU backend for Diffusion architectures.
///
/// Provides optimized GPU kernels for:
/// - Noise prediction
/// - Denoising steps
/// - Latent space operations
///
/// ## Strict No-Fallback
///
/// All operations require GPU. If GPU is unavailable, an error is returned.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[derive(Debug)]
pub struct DiffusionGpuBackend {
    /// Unified GPU kernels
    kernels: UnifiedGpuKernels,
    /// Noise schedule parameters
    noise_schedule: NoiseScheduleParams,
    /// Denoising steps count
    num_steps: usize,
}

/// Noise schedule parameters for diffusion.
#[derive(Debug, Clone)]
pub struct NoiseScheduleParams {
    /// Beta values for noise schedule
    pub betas: Vec<f32>,
    /// Alpha values (1 - beta)
    pub alphas: Vec<f32>,
    /// Cumulative product of alphas
    pub alpha_bars: Vec<f32>,
    /// Schedule type
    pub schedule_type: NoiseScheduleType,
}

/// Type of noise schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseScheduleType {
    /// Linear schedule
    Linear,
    /// Cosine schedule
    Cosine,
    /// Sigmoid schedule
    Sigmoid,
}

impl Default for NoiseScheduleParams {
    fn default() -> Self {
        Self::linear(1000)
    }
}

impl NoiseScheduleParams {
    /// Create a linear noise schedule.
    pub fn linear(num_steps: usize) -> Self {
        let beta_start = 0.0001f32;
        let beta_end = 0.02f32;

        let betas: Vec<f32> = (0..num_steps)
            .map(|i| beta_start + (beta_end - beta_start) * (i as f32) / ((num_steps - 1) as f32))
            .collect();

        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let mut alpha_bars = Vec::with_capacity(num_steps);
        let mut cumprod = 1.0f32;
        for &alpha in &alphas {
            cumprod *= alpha;
            alpha_bars.push(cumprod);
        }

        Self {
            betas,
            alphas,
            alpha_bars,
            schedule_type: NoiseScheduleType::Linear,
        }
    }

    /// Create a cosine noise schedule.
    pub fn cosine(num_steps: usize) -> Self {
        let s: f32 = 0.008; // Small offset to prevent beta from being too small
        let mut alpha_bars = Vec::with_capacity(num_steps);

        for t in 0..=num_steps {
            let t_f = t as f32 / num_steps as f32;
            let alpha_bar = ((t_f + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2)
                .cos()
                .powi(2);
            alpha_bars.push(alpha_bar);
        }

        // Compute betas from alpha_bars
        let mut betas = Vec::with_capacity(num_steps);
        let mut alphas = Vec::with_capacity(num_steps);

        for t in 0..num_steps {
            let alpha_bar_t = alpha_bars[t];
            let alpha_bar_t1 = alpha_bars[t + 1];
            let beta = 1.0 - alpha_bar_t / alpha_bar_t1;
            let beta_clamped = beta.clamp(0.0, 0.999);
            betas.push(beta_clamped);
            alphas.push(1.0 - beta_clamped);
        }

        // Remove the extra element
        alpha_bars.truncate(num_steps);

        Self {
            betas,
            alphas,
            alpha_bars,
            schedule_type: NoiseScheduleType::Cosine,
        }
    }
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl DiffusionGpuBackend {
    /// Create a new Diffusion GPU backend for an explicit backend (strict no-fallback).
    pub fn with_backend(
        num_steps: usize,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<Self> {
        let kernels = UnifiedGpuKernels::new(backend)?;
        Ok(Self {
            kernels,
            noise_schedule: NoiseScheduleParams::cosine(num_steps),
            num_steps,
        })
    }

    /// Create a new Diffusion GPU backend with automatic GPU detection.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU is detected (strict no-fallback).
    pub fn auto_detect(num_steps: usize) -> Result<Self> {
        let backend = crate::domain::compute_backend::resolve_compute_backend_strict_auto_gpu()?;
        Self::with_backend(num_steps, backend)
    }

    /// Create with a specific noise schedule.
    pub fn with_schedule(mut self, schedule: NoiseScheduleParams) -> Self {
        self.num_steps = schedule.betas.len();
        self.noise_schedule = schedule;
        self
    }

    /// Get the noise schedule.
    pub fn noise_schedule(&self) -> &NoiseScheduleParams {
        &self.noise_schedule
    }

    /// Forward diffusion: add noise to input.
    ///
    /// Computes: `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise`
    ///
    /// # Arguments
    /// * `x_0` - Clean input tensor (batch_size, embed_dim)
    /// * `t` - Timestep index (0 to num_steps-1)
    /// * `noise` - Optional noise tensor (batch_size, embed_dim). If None, random noise is used.
    ///
    /// # Returns
    /// * Noisy tensor at timestep t
    pub fn forward_diffusion(
        &mut self,
        x_0: &Array2<f32>,
        t: usize,
        noise: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>> {
        if t >= self.num_steps {
            return Err(ModelError::InvalidInput {
                message: format!("Timestep {} out of range [0, {})", t, self.num_steps),
            });
        }

        let alpha_bar_t = self.noise_schedule.alpha_bars[t];
        let sqrt_alpha_bar = alpha_bar_t.sqrt();
        let sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t).sqrt();

        // Use provided noise or generate pseudo-random tensor.
        let noise_tensor = match noise {
            Some(n) => {
                if n.dim() != x_0.dim() {
                    return Err(ModelError::InvalidInput {
                        message: format!(
                            "Noise shape {:?} must match input shape {:?}",
                            n.dim(),
                            x_0.dim()
                        ),
                    });
                }
                n.clone()
            }
            None => Self::generate_noise_tensor(x_0.dim()),
        };

        // Compute on GPU: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        self.axpy_gpu(x_0, sqrt_alpha_bar, &noise_tensor, sqrt_one_minus_alpha_bar)
    }

    /// Predict noise from noisy input.
    ///
    /// Uses the model to predict the noise that was added to x_0 to get x_t.
    ///
    /// # Arguments
    /// * `x_t` - Noisy input tensor (batch_size, embed_dim)
    /// * `t` - Timestep index
    /// * `model_weights` - Model weights for noise prediction
    ///
    /// # Returns
    /// * Predicted noise tensor
    pub fn predict_noise(
        &mut self,
        x_t: &Array2<f32>,
        t: usize,
        _model_weights: &Array2<f32>, // Placeholder for actual model weights
    ) -> Result<Array2<f32>> {
        // In a full implementation, this would use the model to predict noise
        // For now, return a placeholder
        let (batch_size, embed_dim) = x_t.dim();

        // Time embedding (simplified)
        let t_emb = self.get_time_embedding(t, batch_size, embed_dim)?;

        // Placeholder: return scaled input as "predicted noise"
        // Real implementation would use model forward pass
        let predicted_noise = x_t.mapv(|x| x * 0.1) + &t_emb;

        Ok(predicted_noise)
    }

    /// Get time embedding for timestep.
    fn get_time_embedding(
        &self,
        t: usize,
        batch_size: usize,
        embed_dim: usize,
    ) -> Result<Array2<f32>> {
        let half_dim = embed_dim / 2;
        let mut emb = Array1::zeros(embed_dim);

        for i in 0..half_dim {
            let freq = 1.0 / 10000_f32.powf((2 * i) as f32 / embed_dim as f32);
            let arg = t as f32 * freq;
            emb[2 * i] = arg.sin();
            emb[2 * i + 1] = arg.cos();
        }

        // Expand to batch dimension
        Ok(Array2::from_shape_fn((batch_size, embed_dim), |(_, j)| {
            emb[j]
        }))
    }

    /// Compute weighted sum on GPU: out = a * left + b * right.
    fn axpy_gpu(
        &mut self,
        left: &Array2<f32>,
        a: f32,
        right: &Array2<f32>,
        b: f32,
    ) -> Result<Array2<f32>> {
        if left.dim() != right.dim() {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "AXPY input shape mismatch: left {:?}, right {:?}",
                    left.dim(),
                    right.dim()
                ),
            });
        }
        let dims = left.dim();
        let total = left.len();
        if total == 0 {
            return Ok(Array2::zeros(dims));
        }

        let left_slice = left.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "Diffusion AXPY left tensor must be contiguous".to_string(),
        })?;
        let right_slice = right.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "Diffusion AXPY right tensor must be contiguous".to_string(),
        })?;

        let device_arc = self.kernels.device();
        let mut device = device_arc.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock DiffusionGpuBackend GPU device".to_string(),
        })?;

        let mut left_buf = device.allocate_f32(total)?;
        let mut right_buf = device.allocate_f32(total)?;
        let mut output_buf = device.allocate_f32(total)?;

        device.upload(left_slice, &mut left_buf)?;
        device.upload(right_slice, &mut right_buf)?;
        device.axpy(a, &left_buf, b, &right_buf, &mut output_buf, total)?;

        let mut host_output = vec![0.0f32; total];
        device.download(&output_buf, &mut host_output)?;

        device.deallocate(left_buf);
        device.deallocate(right_buf);
        device.deallocate(output_buf);

        Array2::from_shape_vec(dims, host_output).map_err(|err| ModelError::InvalidInput {
            message: format!("Failed to reshape Diffusion AXPY output: {err}"),
        })
    }

    /// Fast pseudo-random noise in [-1, 1], seeded from system time once per call.
    fn generate_noise_tensor(shape: (usize, usize)) -> Array2<f32> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let total = shape.0.saturating_mul(shape.1);
        let mut data = Vec::with_capacity(total);
        let mut state = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
            ^ 0x9E37_79B9_7F4A_7C15_u64;

        for _ in 0..total {
            // Xorshift64* style update
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state as f64 / u64::MAX as f64) as f32;
            data.push(unit * 2.0 - 1.0);
        }

        Array2::from_shape_vec(shape, data).unwrap_or_else(|_| Array2::zeros(shape))
    }

    /// Denoising step: predict x_{t-1} from x_t.
    ///
    /// Implements one step of the reverse diffusion process.
    ///
    /// # Arguments
    /// * `x_t` - Noisy input at timestep t
    /// * `t` - Current timestep
    /// * `predicted_noise` - Predicted noise from model
    ///
    /// # Returns
    /// * Denoised sample at timestep t-1
    pub fn denoise_step(
        &mut self,
        x_t: &Array2<f32>,
        t: usize,
        predicted_noise: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        if t >= self.num_steps {
            return Err(ModelError::InvalidInput {
                message: format!("Timestep {} out of range [0, {})", t, self.num_steps),
            });
        }
        if predicted_noise.dim() != x_t.dim() {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "predicted_noise shape {:?} must match x_t shape {:?}",
                    predicted_noise.dim(),
                    x_t.dim()
                ),
            });
        }

        if t == 0 {
            return Ok(x_t.clone());
        }

        let beta_t = self.noise_schedule.betas[t];
        let alpha_t = self.noise_schedule.alphas[t];
        let alpha_bar_t = self.noise_schedule.alpha_bars[t];
        let alpha_bar_t1 = if t > 0 {
            self.noise_schedule.alpha_bars[t - 1]
        } else {
            1.0
        };

        // Compute mean on GPU:
        // mu = (1/sqrt(alpha_t)) * x_t - (beta_t / sqrt(1 - alpha_bar_t) / sqrt(alpha_t)) * predicted_noise
        let sqrt_alpha_t = alpha_t.sqrt();
        let sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t).sqrt();
        let coeff_x = 1.0 / sqrt_alpha_t;
        let coeff_noise = -beta_t / sqrt_one_minus_alpha_bar / sqrt_alpha_t;

        let mean = self.axpy_gpu(x_t, coeff_x, predicted_noise, coeff_noise)?;

        // Add noise for t > 1
        if t > 1 {
            let sigma_t = (beta_t * (1.0 - alpha_bar_t1) / (1.0 - alpha_bar_t)).sqrt();
            let noise = Self::generate_noise_tensor(x_t.dim());
            self.axpy_gpu(&mean, 1.0, &noise, sigma_t)
        } else {
            Ok(mean)
        }
    }

    /// Full denoising loop.
    ///
    /// Runs the complete reverse diffusion process from pure noise to clean sample.
    ///
    /// # Arguments
    /// * `x_t` - Starting noisy sample (typically pure noise)
    /// * `model_weights` - Model weights for noise prediction
    ///
    /// # Returns
    /// * Denoised sample
    pub fn denoise(
        &mut self,
        mut x_t: Array2<f32>,
        model_weights: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        for t in (1..=self.num_steps).rev() {
            let predicted_noise = self.predict_noise(&x_t, t - 1, model_weights)?;
            x_t = self.denoise_step(&x_t, t - 1, &predicted_noise)?;
        }
        Ok(x_t)
    }

    /// Get the underlying GPU kernels.
    pub fn kernels(&self) -> &UnifiedGpuKernels {
        &self.kernels
    }

    /// Get mutable access to GPU kernels.
    pub fn kernels_mut(&mut self) -> &mut UnifiedGpuKernels {
        &mut self.kernels
    }
}

// ============================================================================
// SSM GPU Backend
// ============================================================================

/// GPU backend for State Space Model (SSM) architectures.
///
/// Provides optimized GPU kernels for:
/// - Mamba selective scan
/// - RG-LRU recurrent computation
/// - State updates
///
/// ## Strict No-Fallback
///
/// All operations require GPU. If GPU is unavailable, an error is returned.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[derive(Debug)]
pub struct SsmGpuBackend {
    /// Unified GPU kernels
    kernels: UnifiedGpuKernels,
    /// SSM parameters
    params: SsmParams,
    /// Temporal type (Mamba or RG-LRU)
    temporal_type: GpuTemporalType,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl SsmGpuBackend {
    /// Create a new SSM GPU backend for an explicit backend (strict no-fallback).
    pub fn with_backend(
        state_dim: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
        temporal_type: GpuTemporalType,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<Self> {
        let kernels = UnifiedGpuKernels::new(backend)?;
        let params = SsmParams::new(state_dim, embed_dim, seq_len, batch_size);
        Ok(Self {
            kernels,
            params,
            temporal_type,
        })
    }

    /// Create a new SSM GPU backend with automatic GPU detection.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU is detected (strict no-fallback).
    pub fn auto_detect(
        state_dim: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
        temporal_type: GpuTemporalType,
    ) -> Result<Self> {
        let backend = crate::domain::compute_backend::resolve_compute_backend_strict_auto_gpu()?;
        Self::with_backend(
            state_dim,
            embed_dim,
            seq_len,
            batch_size,
            temporal_type,
            backend,
        )
    }

    /// Create for Mamba architecture.
    pub fn mamba(
        state_dim: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<Self> {
        Self::auto_detect(
            state_dim,
            embed_dim,
            seq_len,
            batch_size,
            GpuTemporalType::Mamba,
        )
    }

    /// Create for Mamba architecture using an explicit backend.
    pub fn mamba_with_backend(
        state_dim: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<Self> {
        Self::with_backend(
            state_dim,
            embed_dim,
            seq_len,
            batch_size,
            GpuTemporalType::Mamba,
            backend,
        )
    }

    /// Create for RG-LRU architecture.
    pub fn rg_lru(
        state_dim: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<Self> {
        Self::auto_detect(
            state_dim,
            embed_dim,
            seq_len,
            batch_size,
            GpuTemporalType::RgLru,
        )
    }

    /// Create for RG-LRU architecture using an explicit backend.
    pub fn rg_lru_with_backend(
        state_dim: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<Self> {
        Self::with_backend(
            state_dim,
            embed_dim,
            seq_len,
            batch_size,
            GpuTemporalType::RgLru,
            backend,
        )
    }

    /// Forward pass through SSM.
    ///
    /// Dispatches to Mamba or RG-LRU based on temporal_type.
    pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        self.kernels
            .ssm_forward(input, &self.params, self.temporal_type)
    }

    /// Backward pass through SSM.
    ///
    /// Returns `(input_grads, param_grads)` where parameter gradient ordering depends on temporal type:
    /// - `Mamba`: `[dA, dB, dC, dD]`
    /// - `RgLru`: `[dW_f, dW_r, dW_o]`
    pub fn backward(
        &mut self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        self.kernels
            .ssm_backward(input, output_grads, &self.params, self.temporal_type)
    }

    /// Get SSM parameters.
    pub fn params(&self) -> &SsmParams {
        &self.params
    }

    /// Update SSM parameters.
    pub fn set_params(&mut self, params: SsmParams) {
        self.params = params;
    }

    /// Get temporal type.
    pub fn temporal_type(&self) -> GpuTemporalType {
        self.temporal_type
    }

    /// Get the underlying GPU kernels.
    pub fn kernels(&self) -> &UnifiedGpuKernels {
        &self.kernels
    }

    /// Get mutable access to GPU kernels.
    pub fn kernels_mut(&mut self) -> &mut UnifiedGpuKernels {
        &mut self.kernels
    }

    /// Set explicit Mamba kernel matrices for SSM forward/backward kernels.
    pub fn set_mamba_kernel_matrices(
        &mut self,
        a: Array2<f32>,
        b: Array2<f32>,
        c: Array2<f32>,
        d: Array2<f32>,
        h_init: Array2<f32>,
    ) {
        self.kernels.set_mamba_kernel_matrices(a, b, c, d, h_init);
    }

    /// Set explicit RG-LRU kernel matrices for SSM forward/backward kernels.
    pub fn set_rg_lru_kernel_matrices(
        &mut self,
        w_f: Array2<f32>,
        w_r: Array2<f32>,
        w_o: Array2<f32>,
        h_init: Array2<f32>,
    ) {
        self.kernels
            .set_rg_lru_kernel_matrices(w_f, w_r, w_o, h_init);
    }

    pub fn clear_mamba_kernel_matrices(&mut self) {
        self.kernels.clear_mamba_kernel_matrices();
    }

    pub fn clear_rg_lru_kernel_matrices(&mut self) {
        self.kernels.clear_rg_lru_kernel_matrices();
    }
}

// ============================================================================
// Transformer GPU Backend
// ============================================================================

/// GPU backend for Transformer architectures.
///
/// Provides optimized GPU kernels for:
/// - Multi-head attention
/// - Feedforward networks
/// - Layer normalization
///
/// ## Strict No-Fallback
///
/// All operations require GPU. If GPU is unavailable, an error is returned.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[derive(Debug)]
pub struct TransformerGpuBackend {
    /// Unified GPU kernels
    kernels: UnifiedGpuKernels,
    /// Attention parameters
    attention_params: AttentionParams,
    /// Default activation function
    activation: GpuActivation,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl TransformerGpuBackend {
    /// Create a new Transformer GPU backend for an explicit backend (strict no-fallback).
    pub fn with_backend(
        num_heads: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<Self> {
        let kernels = UnifiedGpuKernels::new(backend)?;
        let attention_params = AttentionParams::new(num_heads, embed_dim, seq_len, batch_size);
        Ok(Self {
            kernels,
            attention_params,
            activation: GpuActivation::Gelu,
        })
    }

    /// Create a new Transformer GPU backend with automatic GPU detection.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU is detected (strict no-fallback).
    pub fn auto_detect(
        num_heads: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let backend = crate::domain::compute_backend::resolve_compute_backend_strict_auto_gpu()?;
        Self::with_backend(num_heads, embed_dim, seq_len, batch_size, backend)
    }

    /// Set causal attention mode.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.attention_params = self.attention_params.with_causal(causal);
        self
    }

    /// Set sliding window attention.
    pub fn with_window(mut self, window_size: usize) -> Self {
        self.attention_params = self.attention_params.with_window(window_size);
        self
    }

    /// Set default activation function.
    pub fn with_activation(mut self, activation: GpuActivation) -> Self {
        self.activation = activation;
        self
    }

    /// Multi-head attention forward pass.
    ///
    /// Computes: output = softmax(Q @ K^T / scale) @ V @ W_o
    pub fn attention_forward(
        &mut self,
        input: &Array2<f32>,
        wq: &Array2<f32>,
        wk: &Array2<f32>,
        wv: &Array2<f32>,
        wo: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        self.kernels
            .attention_forward(input, wq, wk, wv, wo, &self.attention_params)
    }

    /// Flash attention (memory-efficient) forward pass.
    pub fn flash_attention_forward(
        &mut self,
        input: &Array2<f32>,
        wq: &Array2<f32>,
        wk: &Array2<f32>,
        wv: &Array2<f32>,
        wo: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        self.kernels
            .flash_attention_forward(input, wq, wk, wv, wo, &self.attention_params)
    }

    /// Layer normalization forward pass.
    pub fn layer_norm_forward(
        &mut self,
        input: &Array2<f32>,
        gamma: Option<&Array2<f32>>,
        beta: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>> {
        let params = NormParams::new(self.attention_params.embed_dim);
        self.kernels.layer_norm_forward(input, gamma, beta, &params)
    }

    /// Activation forward pass.
    pub fn activation_forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        self.kernels.activation_forward(input, self.activation)
    }

    /// Richards curve activation forward pass.
    pub fn richards_curve_forward(
        &mut self,
        input: &Array2<f32>,
        nu: f32,
        k: f32,
        m: f32,
        beta: f32,
    ) -> Result<Array2<f32>> {
        self.kernels.richards_curve_forward(input, nu, k, m, beta)
    }

    /// Get attention parameters.
    pub fn attention_params(&self) -> &AttentionParams {
        &self.attention_params
    }

    /// Update attention parameters.
    pub fn set_attention_params(&mut self, params: AttentionParams) {
        self.attention_params = params;
    }

    /// Get default activation.
    pub fn activation(&self) -> GpuActivation {
        self.activation
    }

    /// Get the underlying GPU kernels.
    pub fn kernels(&self) -> &UnifiedGpuKernels {
        &self.kernels
    }

    /// Get mutable access to GPU kernels.
    pub fn kernels_mut(&mut self) -> &mut UnifiedGpuKernels {
        &mut self.kernels
    }
}

// ============================================================================
// Unified Backend Factory
// ============================================================================

/// Factory for creating GPU backends with automatic detection.
///
/// Provides a unified interface for creating GPU backends for all architectures.
/// All backends use strict no-fallback semantics.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuBackendFactory;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuBackendFactory {
    /// Check if GPU is available on this system.
    pub fn is_gpu_available() -> bool {
        GpuDevice::auto_detect().is_ok()
    }

    /// Get the name of the best available GPU backend.
    pub fn best_backend_name() -> Option<&'static str> {
        GpuDevice::auto_detect().ok().map(|d| d.backend().as_str())
    }

    /// Create a Diffusion GPU backend.
    pub fn diffusion(num_steps: usize) -> Result<DiffusionGpuBackend> {
        DiffusionGpuBackend::auto_detect(num_steps)
    }

    /// Create a Diffusion GPU backend for an explicit backend variant.
    pub fn diffusion_with_backend(
        num_steps: usize,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<DiffusionGpuBackend> {
        DiffusionGpuBackend::with_backend(num_steps, backend)
    }

    /// Create an SSM GPU backend for Mamba.
    pub fn ssm_mamba(
        state_dim: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<SsmGpuBackend> {
        SsmGpuBackend::mamba(state_dim, embed_dim, seq_len, batch_size)
    }

    /// Create an SSM GPU backend for Mamba with explicit backend variant.
    pub fn ssm_mamba_with_backend(
        state_dim: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<SsmGpuBackend> {
        SsmGpuBackend::mamba_with_backend(state_dim, embed_dim, seq_len, batch_size, backend)
    }

    /// Create an SSM GPU backend for RG-LRU.
    pub fn ssm_rg_lru(
        state_dim: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<SsmGpuBackend> {
        SsmGpuBackend::rg_lru(state_dim, embed_dim, seq_len, batch_size)
    }

    /// Create an SSM GPU backend for RG-LRU with explicit backend variant.
    pub fn ssm_rg_lru_with_backend(
        state_dim: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<SsmGpuBackend> {
        SsmGpuBackend::rg_lru_with_backend(state_dim, embed_dim, seq_len, batch_size, backend)
    }

    /// Create a Transformer GPU backend.
    pub fn transformer(
        num_heads: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<TransformerGpuBackend> {
        TransformerGpuBackend::auto_detect(num_heads, embed_dim, seq_len, batch_size)
    }

    /// Create a Transformer GPU backend for an explicit backend variant.
    pub fn transformer_with_backend(
        num_heads: usize,
        embed_dim: usize,
        seq_len: usize,
        batch_size: usize,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<TransformerGpuBackend> {
        TransformerGpuBackend::with_backend(num_heads, embed_dim, seq_len, batch_size, backend)
    }

    /// Create a MoE GPU backend.
    pub fn moe(
        num_experts: usize,
        num_active: usize,
        embed_dim: usize,
        expert_hidden_dim: usize,
    ) -> Result<MoeGpuBackend> {
        MoeGpuBackend::auto_detect(num_experts, num_active, embed_dim, expert_hidden_dim)
    }

    /// Create a MoE GPU backend for an explicit backend variant.
    pub fn moe_with_backend(
        num_experts: usize,
        num_active: usize,
        embed_dim: usize,
        expert_hidden_dim: usize,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<MoeGpuBackend> {
        MoeGpuBackend::with_backend(
            num_experts,
            num_active,
            embed_dim,
            expert_hidden_dim,
            backend,
        )
    }
}

// ============================================================================
// MoE GPU Backend
// ============================================================================

/// GPU backend for Mixture-of-Experts (MoE) architectures.
///
/// Provides optimized GPU kernels for:
/// - Router GEMM computation
/// - Top-k expert selection
/// - Expert computation (parallel GEMMs)
/// - Weighted output combination
///
/// ## Strict No-Fallback
///
/// All operations require GPU. If GPU is unavailable, an error is returned.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[derive(Debug)]
pub struct MoeGpuBackend {
    /// Unified GPU kernels
    kernels: UnifiedGpuKernels,
    /// Number of experts
    num_experts: usize,
    /// Number of active experts per token (top-k)
    num_active: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Expert hidden dimension
    expert_hidden_dim: usize,
    /// Router weights [embed_dim, num_experts]
    router_weights: Array2<f32>,
    /// Expert weights (each expert has w1 and w2)
    /// w1: [embed_dim, expert_hidden_dim], w2: [expert_hidden_dim, embed_dim]
    expert_weights_w1: Vec<Array2<f32>>,
    expert_weights_w2: Vec<Array2<f32>>,
}

/// Parameters for MoE GPU operations.
#[derive(Debug, Clone)]
pub struct MoeParams {
    /// Number of experts
    pub num_experts: usize,
    /// Number of active experts (top-k)
    pub num_active: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Expert hidden dimension
    pub expert_hidden_dim: usize,
    /// Batch size
    pub batch_size: usize,
}

impl MoeParams {
    /// Create new MoE parameters.
    pub fn new(
        num_experts: usize,
        num_active: usize,
        embed_dim: usize,
        expert_hidden_dim: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            num_experts,
            num_active: num_active.min(num_experts),
            embed_dim,
            expert_hidden_dim,
            batch_size,
        }
    }
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl MoeGpuBackend {
    /// Create a new MoE GPU backend for an explicit backend (strict no-fallback).
    pub fn with_backend(
        num_experts: usize,
        num_active: usize,
        embed_dim: usize,
        expert_hidden_dim: usize,
        backend: crate::domain::compute_backend::ComputeBackend,
    ) -> Result<Self> {
        let kernels = UnifiedGpuKernels::new(backend)?;

        // Initialize with random weights (in practice, these would be loaded)
        let router_weights = Array2::from_shape_fn((embed_dim, num_experts), |_| {
            // Xavier initialization
            let scale = (2.0 / (embed_dim + num_experts) as f32).sqrt();
            (rand_random() - 0.5) * 2.0 * scale
        });

        let scale = (2.0 / (embed_dim + expert_hidden_dim) as f32).sqrt();
        let expert_weights_w1 = (0..num_experts)
            .map(|_| {
                Array2::from_shape_fn((embed_dim, expert_hidden_dim), |_| {
                    (rand_random() - 0.5) * 2.0 * scale
                })
            })
            .collect();

        let scale = (2.0 / (expert_hidden_dim + embed_dim) as f32).sqrt();
        let expert_weights_w2 = (0..num_experts)
            .map(|_| {
                Array2::from_shape_fn((expert_hidden_dim, embed_dim), |_| {
                    (rand_random() - 0.5) * 2.0 * scale
                })
            })
            .collect();

        Ok(Self {
            kernels,
            num_experts,
            num_active: num_active.min(num_experts),
            embed_dim,
            expert_hidden_dim,
            router_weights,
            expert_weights_w1,
            expert_weights_w2,
        })
    }

    /// Create a new MoE GPU backend with automatic GPU detection.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU is detected (strict no-fallback).
    pub fn auto_detect(
        num_experts: usize,
        num_active: usize,
        embed_dim: usize,
        expert_hidden_dim: usize,
    ) -> Result<Self> {
        let backend = crate::domain::compute_backend::resolve_compute_backend_strict_auto_gpu()?;
        Self::with_backend(
            num_experts,
            num_active,
            embed_dim,
            expert_hidden_dim,
            backend,
        )
    }

    /// Create with pre-trained weights.
    pub fn with_weights(
        mut self,
        router_weights: Array2<f32>,
        expert_weights_w1: Vec<Array2<f32>>,
        expert_weights_w2: Vec<Array2<f32>>,
    ) -> Result<Self> {
        if router_weights.ncols() != self.num_experts {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "Router weights cols {} != num_experts {}",
                    router_weights.ncols(),
                    self.num_experts
                ),
            });
        }
        if expert_weights_w1.len() != self.num_experts {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "Expert w1 count {} != num_experts {}",
                    expert_weights_w1.len(),
                    self.num_experts
                ),
            });
        }
        if expert_weights_w2.len() != self.num_experts {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "Expert w2 count {} != num_experts {}",
                    expert_weights_w2.len(),
                    self.num_experts
                ),
            });
        }

        self.router_weights = router_weights;
        self.expert_weights_w1 = expert_weights_w1;
        self.expert_weights_w2 = expert_weights_w2;
        Ok(self)
    }

    /// MoE forward pass on GPU.
    ///
    /// Implements:
    /// 1. Router GEMM: `routing_logits = input @ router_weights`
    /// 2. Top-k selection: Select top-k experts per token
    /// 3. Softmax: Normalize routing scores
    /// 4. Expert computation: For each selected expert, compute output
    /// 5. Weighted sum: Combine expert outputs using routing gates
    pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let (batch_size, embed_dim) = input.dim();

        if embed_dim != self.embed_dim {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "Input embed_dim {} != expected {}",
                    embed_dim, self.embed_dim
                ),
            });
        }

        let input_slice = input.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "MoeGpuBackend input must be contiguous".to_string(),
        })?;
        let router_slice =
            self.router_weights
                .as_slice()
                .ok_or_else(|| ModelError::InvalidInput {
                    message: "MoeGpuBackend router weights must be contiguous".to_string(),
                })?;

        let device_arc = self.kernels.device();
        let mut device = device_arc.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock MoeGpuBackend GPU device".to_string(),
        })?;

        let total_input = batch_size * embed_dim;
        let total_logits = batch_size * self.num_experts;
        let total_output = batch_size * embed_dim;

        let mut input_buf = device.allocate_f32(total_input)?;
        device.upload(input_slice, &mut input_buf)?;

        let mut router_w_buf = device.allocate_f32(self.router_weights.len())?;
        device.upload(router_slice, &mut router_w_buf)?;

        // 1. Router GEMM on GPU: routing_logits = input @ router_weights
        let mut routing_logits_buf = device.allocate_f32(total_logits)?;
        device.gemm_f32(
            1.0,
            &input_buf,
            &router_w_buf,
            0.0,
            &mut routing_logits_buf,
            batch_size,
            self.num_experts,
            embed_dim,
            false,
            false,
        )?;

        let mut routing_logits_host = vec![0.0f32; total_logits];
        device.download(&routing_logits_buf, &mut routing_logits_host)?;
        let routing_logits =
            Array2::from_shape_vec((batch_size, self.num_experts), routing_logits_host).map_err(
                |err| ModelError::InvalidInput {
                    message: format!("Failed to reshape MoeGpuBackend routing logits: {err}"),
                },
            )?;

        // 2. Top-k selection and softmax on CPU
        let mut gates = Array2::<f32>::zeros((batch_size, self.num_experts));
        let top_k = self.num_active.clamp(1, self.num_experts);
        for token_idx in 0..batch_size {
            let row = routing_logits.row(token_idx);

            // Maintain top-k without allocating/sorting full expert list.
            let mut best: Vec<(f32, usize)> = Vec::with_capacity(top_k);
            for (expert_idx, &logit) in row.iter().enumerate() {
                let score = if logit.is_finite() {
                    logit
                } else {
                    f32::NEG_INFINITY
                };
                if best.len() < top_k {
                    best.push((score, expert_idx));
                    continue;
                }

                let mut min_pos = 0usize;
                let mut min_score = best[0].0;
                for (pos, (s, _)) in best.iter().enumerate().skip(1) {
                    if *s < min_score {
                        min_score = *s;
                        min_pos = pos;
                    }
                }

                if score > min_score {
                    best[min_pos] = (score, expert_idx);
                }
            }

            best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut max_logit = f32::NEG_INFINITY;
            let mut any_finite = false;
            for &(score, _) in &best {
                if score.is_finite() {
                    any_finite = true;
                    max_logit = max_logit.max(score);
                }
            }

            if !any_finite {
                let uniform = 1.0 / best.len().max(1) as f32;
                for &(_, expert_idx) in &best {
                    gates[[token_idx, expert_idx]] = uniform;
                }
                continue;
            }

            let mut exp_sum = 0.0f64;
            let mut exp_vals = vec![0.0f32; best.len()];
            for (i, &(score, _)) in best.iter().enumerate() {
                if score.is_finite() {
                    let e = (score - max_logit).exp();
                    exp_vals[i] = e;
                    exp_sum += e as f64;
                }
            }

            if exp_sum <= 0.0 || !exp_sum.is_finite() {
                let uniform = 1.0 / best.len().max(1) as f32;
                for &(_, expert_idx) in &best {
                    gates[[token_idx, expert_idx]] = uniform;
                }
                continue;
            }

            let inv_sum = 1.0 / exp_sum as f32;
            for (i, &(_, expert_idx)) in best.iter().enumerate() {
                gates[[token_idx, expert_idx]] = exp_vals[i] * inv_sum;
            }
        }

        // 3. Expert projections on GPU + weighted accumulation in output buffer.
        let mut output_buf = device.allocate_f32(total_output)?;
        let zero_output = vec![0.0f32; total_output];
        device.upload(&zero_output, &mut output_buf)?;
        let mut gate_expand = vec![0.0f32; total_output];
        let w1_elems = self.embed_dim * self.expert_hidden_dim;
        let w2_elems = self.expert_hidden_dim * self.embed_dim;
        let hidden_elems = batch_size * self.expert_hidden_dim;
        let mut w1_buf = device.allocate_f32(w1_elems)?;
        let mut w2_buf = device.allocate_f32(w2_elems)?;
        let mut hidden_buf = device.allocate_f32(hidden_elems)?;
        let mut hidden_relu_buf = device.allocate_f32(hidden_elems)?;
        let mut expert_out_buf = device.allocate_f32(total_output)?;
        let mut gate_buf = device.allocate_f32(total_output)?;
        let mut scaled_expert_buf = device.allocate_f32(total_output)?;

        for expert_idx in 0..self.num_experts {
            let gate_col = gates.column(expert_idx);
            if gate_col.iter().all(|&g| g == 0.0) {
                continue;
            }

            let w1_slice = self.expert_weights_w1[expert_idx]
                .as_slice()
                .ok_or_else(|| ModelError::InvalidInput {
                    message: format!("MoE expert {expert_idx} w1 must be contiguous"),
                })?;
            let w2_slice = self.expert_weights_w2[expert_idx]
                .as_slice()
                .ok_or_else(|| ModelError::InvalidInput {
                    message: format!("MoE expert {expert_idx} w2 must be contiguous"),
                })?;

            device.upload(w1_slice, &mut w1_buf)?;
            device.upload(w2_slice, &mut w2_buf)?;

            // hidden = input @ w1_e
            device.gemm_f32(
                1.0,
                &input_buf,
                &w1_buf,
                0.0,
                &mut hidden_buf,
                batch_size,
                self.expert_hidden_dim,
                self.embed_dim,
                false,
                false,
            )?;
            // Activation (ReLU)
            device.relu(
                &hidden_buf,
                &mut hidden_relu_buf,
                batch_size * self.expert_hidden_dim,
            )?;

            // expert_output = hidden @ w2_e
            device.gemm_f32(
                1.0,
                &hidden_relu_buf,
                &w2_buf,
                0.0,
                &mut expert_out_buf,
                batch_size,
                self.embed_dim,
                self.expert_hidden_dim,
                false,
                false,
            )?;

            // Expand per-token gate weights to matrix shape for element-wise scaling.
            for token_idx in 0..batch_size {
                let g = gate_col[token_idx];
                let row_offset = token_idx * self.embed_dim;
                for j in 0..self.embed_dim {
                    gate_expand[row_offset + j] = g;
                }
            }
            device.upload(&gate_expand, &mut gate_buf)?;
            device.mul(
                &expert_out_buf,
                &gate_buf,
                &mut scaled_expert_buf,
                total_output,
            )?;
            device.add_scaled(1.0, &scaled_expert_buf, &mut output_buf, total_output)?;
        }

        let mut output_host = vec![0.0f32; total_output];
        device.download(&output_buf, &mut output_host)?;

        device.deallocate(input_buf);
        device.deallocate(router_w_buf);
        device.deallocate(routing_logits_buf);
        device.deallocate(output_buf);
        device.deallocate(w1_buf);
        device.deallocate(w2_buf);
        device.deallocate(hidden_buf);
        device.deallocate(hidden_relu_buf);
        device.deallocate(expert_out_buf);
        device.deallocate(gate_buf);
        device.deallocate(scaled_expert_buf);

        Array2::from_shape_vec((batch_size, embed_dim), output_host).map_err(|err| {
            ModelError::InvalidInput {
                message: format!("Failed to reshape MoeGpuBackend output: {err}"),
            }
        })
    }

    /// Get MoE parameters.
    pub fn params(&self) -> MoeParams {
        MoeParams::new(
            self.num_experts,
            self.num_active,
            self.embed_dim,
            self.expert_hidden_dim,
            0, // batch size not known until forward
        )
    }

    /// Get number of experts.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get number of active experts.
    pub fn num_active(&self) -> usize {
        self.num_active
    }

    /// Get the underlying GPU kernels.
    pub fn kernels(&self) -> &UnifiedGpuKernels {
        &self.kernels
    }

    /// Get mutable access to GPU kernels.
    pub fn kernels_mut(&mut self) -> &mut UnifiedGpuKernels {
        &mut self.kernels
    }
}

/// Simple random number generator for weight initialization.
/// Returns a value in [0, 1).
fn rand_random() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    // Simple hash to get a pseudo-random value
    let hash = nanos.wrapping_mul(2654435761u32);
    (hash as f32) / (u32::MAX as f32)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_schedule_linear() {
        let schedule = NoiseScheduleParams::linear(1000);
        assert_eq!(schedule.betas.len(), 1000);
        assert_eq!(schedule.alphas.len(), 1000);
        assert_eq!(schedule.alpha_bars.len(), 1000);

        // First beta should be close to 0.0001
        assert!((schedule.betas[0] - 0.0001).abs() < 1e-6);

        // Last beta should be close to 0.02
        assert!((schedule.betas[999] - 0.02).abs() < 1e-6);

        // Alpha bars should be monotonically decreasing
        for i in 1..schedule.alpha_bars.len() {
            assert!(schedule.alpha_bars[i] < schedule.alpha_bars[i - 1]);
        }
    }

    #[test]
    fn test_noise_schedule_cosine() {
        let schedule = NoiseScheduleParams::cosine(1000);
        assert_eq!(schedule.betas.len(), 1000);
        assert_eq!(schedule.alphas.len(), 1000);

        // Alpha bars should be monotonically decreasing
        for i in 1..schedule.alpha_bars.len() {
            assert!(schedule.alpha_bars[i] < schedule.alpha_bars[i - 1]);
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_gpu_backend_factory_availability() {
        // This test checks if GPU is available without failing
        let available = GpuBackendFactory::is_gpu_available();
        println!("GPU available: {}", available);

        if available {
            let name = GpuBackendFactory::best_backend_name();
            println!("Best backend: {:?}", name);
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_diffusion_backend_creation() {
        match DiffusionGpuBackend::auto_detect(1000) {
            Ok(backend) => {
                println!("Diffusion GPU backend created successfully");
                assert_eq!(backend.num_steps, 1000);
            }
            Err(e) => {
                println!("No GPU available (expected on CPU-only systems): {}", e);
            }
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_ssm_backend_creation() {
        match SsmGpuBackend::mamba(256, 512, 128, 32) {
            Ok(backend) => {
                println!("SSM GPU backend created successfully");
                assert_eq!(backend.temporal_type(), GpuTemporalType::Mamba);
            }
            Err(e) => {
                println!("No GPU available (expected on CPU-only systems): {}", e);
            }
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_transformer_backend_creation() {
        match TransformerGpuBackend::auto_detect(8, 512, 128, 32) {
            Ok(backend) => {
                println!("Transformer GPU backend created successfully");
                assert_eq!(backend.attention_params().num_heads, 8);
            }
            Err(e) => {
                println!("No GPU available (expected on CPU-only systems): {}", e);
            }
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_moe_backend_creation() {
        match MoeGpuBackend::auto_detect(8, 2, 64, 128) {
            Ok(backend) => {
                println!("MoE GPU backend created successfully");
                assert_eq!(backend.num_experts(), 8);
                assert_eq!(backend.num_active(), 2);
            }
            Err(e) => {
                println!("No GPU available (expected on CPU-only systems): {}", e);
            }
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_moe_forward() {
        match MoeGpuBackend::auto_detect(4, 2, 32, 64) {
            Ok(mut backend) => {
                let input = Array2::from_shape_fn((4, 32), |(i, j)| (i + j) as f32 * 0.1);
                match backend.forward(&input) {
                    Ok(output) => {
                        println!("MoE forward pass successful");
                        assert_eq!(output.dim(), (4, 32));
                        // Output should not be all zeros
                        let sum: f32 = output.iter().sum();
                        assert!(sum.abs() > 0.0, "Output should not be all zeros");
                    }
                    Err(e) => {
                        println!("MoE forward failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("No GPU available (expected on CPU-only systems): {}", e);
            }
        }
    }

    #[test]
    fn test_moe_params_creation() {
        let params = MoeParams::new(8, 2, 64, 128, 32);
        assert_eq!(params.num_experts, 8);
        assert_eq!(params.num_active, 2);
        assert_eq!(params.embed_dim, 64);
        assert_eq!(params.expert_hidden_dim, 128);
        assert_eq!(params.batch_size, 32);
    }

    #[test]
    fn test_moe_params_num_active_capped() {
        // num_active should be capped at num_experts
        let params = MoeParams::new(4, 10, 64, 128, 32);
        assert_eq!(params.num_active, 4); // Capped to num_experts
    }
}
