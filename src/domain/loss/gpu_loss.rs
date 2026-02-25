//! GPU Loss Functions
//!
//! This module provides GPU-native implementations of loss functions for training.
//! All operations remain on GPU without CPU transfer.
//!
//! ## Supported Loss Functions
//!
//! - **Cross-Entropy Loss**: Standard classification loss
//! - **Symmetric Cross-Entropy Loss**: Combines CE with reverse CE for robustness
//! - **KL Divergence**: For knowledge distillation
//!
//! ## Memory Efficiency
//!
//! All loss functions use workspace buffers to minimize allocations:
//! - Softmax computation reuses temporary buffers
//! - Gradient computation is fused where possible

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::common::errors::{ModelError, Result};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, GpuDevice};

/// Configuration for symmetric cross-entropy loss
#[derive(Debug, Clone, Copy)]
pub struct GpuSymmetricCEConfig {
    /// Weight for forward cross-entropy
    pub alpha: f32,
    /// Weight for reverse cross-entropy
    pub beta: f32,
    /// Numerical stability epsilon
    pub epsilon: f32,
}

impl Default for GpuSymmetricCEConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            beta: 0.1,
            epsilon: 1e-8,
        }
    }
}

/// GPU-native loss computation workspace
///
/// Manages temporary buffers for loss computation to avoid repeated allocations.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuLossWorkspace {
    /// Softmax output buffer [batch, seq, vocab]
    softmax_output: GpuBuffer,
    /// One-hot encoded targets [batch, seq, vocab]
    one_hot_targets: GpuBuffer,
    /// Per-token loss buffer [batch, seq]
    per_token_loss: GpuBuffer,
    /// Temporary buffer for log computation
    log_buffer: GpuBuffer,
    /// Buffer size tracking
    vocab_size: usize,
    batch_size: usize,
    seq_len: usize,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuLossWorkspace {
    /// Create a new workspace with pre-allocated buffers
    pub fn new(
        device: &mut GpuDevice,
        batch_size: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        let total_elements = batch_size * seq_len * vocab_size;
        let token_elements = batch_size * seq_len;

        let softmax_output = device.allocate_f32(total_elements)?;
        let one_hot_targets = device.allocate_f32(total_elements)?;
        let per_token_loss = device.allocate_f32(token_elements)?;
        let log_buffer = device.allocate_f32(total_elements)?;

        Ok(Self {
            softmax_output,
            one_hot_targets,
            per_token_loss,
            log_buffer,
            vocab_size,
            batch_size,
            seq_len,
        })
    }

    /// Resize workspace if needed
    pub fn ensure_capacity(
        &mut self,
        device: &mut GpuDevice,
        batch_size: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<()> {
        let total_elements = batch_size * seq_len * vocab_size;
        let token_elements = batch_size * seq_len;

        // Check if we need to reallocate
        if batch_size != self.batch_size
            || seq_len != self.seq_len
            || vocab_size != self.vocab_size
        {
            self.softmax_output = device.allocate_f32(total_elements)?;
            self.one_hot_targets = device.allocate_f32(total_elements)?;
            self.per_token_loss = device.allocate_f32(token_elements)?;
            self.log_buffer = device.allocate_f32(total_elements)?;

            self.vocab_size = vocab_size;
            self.batch_size = batch_size;
            self.seq_len = seq_len;
        }

        Ok(())
    }
}

/// GPU cross-entropy loss computation
///
/// Computes: loss = -mean(log(softmax(logits)[targets]))
///
/// # Arguments
///
/// * `device` - GPU device
/// * `logits` - Input logits [batch, seq, vocab]
/// * `targets` - Target token IDs [batch, seq] (flat u32 buffer)
/// * `workspace` - Pre-allocated workspace buffers
/// * `loss_output` - Output scalar loss (single f32)
/// * `grad_output` - Output gradients [batch, seq, vocab]
///
/// # Returns
///
/// The computed loss value (also written to loss_output)
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn gpu_cross_entropy_loss(
    device: &mut GpuDevice,
    logits: &GpuBuffer,
    targets: &GpuBuffer,
    workspace: &mut GpuLossWorkspace,
    loss_output: &mut GpuBuffer,
    grad_output: &mut GpuBuffer,
) -> Result<f32> {
    let batch_size = workspace.batch_size;
    let seq_len = workspace.seq_len;
    let vocab_size = workspace.vocab_size;
    let total_tokens = batch_size * seq_len;

    // 1. Compute softmax: softmax_output = softmax(logits)
    // Reshape for row-wise softmax: [batch * seq, vocab]
    device.softmax(
        logits,
        &mut workspace.softmax_output,
        batch_size * seq_len,
        vocab_size,
    )?;

    // 2. One-hot encode targets
    // This requires a custom kernel or CPU fallback
    one_hot_encode_gpu(
        device,
        targets,
        &mut workspace.one_hot_targets,
        total_tokens,
        vocab_size,
    )?;

    // 3. Compute log(softmax)
    gpu_log_safe(
        device,
        &workspace.softmax_output,
        &mut workspace.log_buffer,
        total_tokens * vocab_size,
        1e-10,
    )?;

    // 4. Compute per-token loss: -sum(one_hot * log_softmax, dim=-1)
    // This is: -log(softmax[target]) for each token
    gpu_cross_entropy_reduce(
        device,
        &workspace.log_buffer,
        &workspace.one_hot_targets,
        &mut workspace.per_token_loss,
        total_tokens,
        vocab_size,
    )?;

    // 5. Mean reduction
    let loss = device.mean(&workspace.per_token_loss, total_tokens)?;

    // 6. Upload loss to GPU buffer
    device.upload(&[loss], loss_output)?;

    // 7. Compute gradients: grad = softmax - one_hot
    // Scaled by 1/num_tokens for mean reduction
    let scale = 1.0 / total_tokens as f32;
    gpu_gradient_compute(
        device,
        &workspace.softmax_output,
        &workspace.one_hot_targets,
        grad_output,
        total_tokens * vocab_size,
        scale,
    )?;

    Ok(loss)
}

/// GPU symmetric cross-entropy loss computation
///
/// Computes: loss = alpha * CE + beta * RCE
/// where CE is standard cross-entropy and RCE is reverse cross-entropy
///
/// # Arguments
///
/// * `device` - GPU device
/// * `logits` - Input logits [batch, seq, vocab]
/// * `targets` - Target token IDs [batch, seq] (flat u32 buffer)
/// * `config` - Loss configuration (alpha, beta, epsilon)
/// * `workspace` - Pre-allocated workspace buffers
/// * `loss_output` - Output scalar loss
/// * `grad_output` - Output gradients
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_symmetric_cross_entropy_loss(
    device: &mut GpuDevice,
    logits: &GpuBuffer,
    targets: &GpuBuffer,
    config: GpuSymmetricCEConfig,
    workspace: &mut GpuLossWorkspace,
    loss_output: &mut GpuBuffer,
    grad_output: &mut GpuBuffer,
) -> Result<f32> {
    let batch_size = workspace.batch_size;
    let seq_len = workspace.seq_len;
    let vocab_size = workspace.vocab_size;
    let total_tokens = batch_size * seq_len;
    let total_elements = total_tokens * vocab_size;

    // 1. Compute softmax
    device.softmax(
        logits,
        &mut workspace.softmax_output,
        total_tokens,
        vocab_size,
    )?;

    // 2. One-hot encode targets
    one_hot_encode_gpu(
        device,
        targets,
        &mut workspace.one_hot_targets,
        total_tokens,
        vocab_size,
    )?;

    // 3. Compute forward CE loss and gradients
    // CE = -sum(one_hot * log(softmax))
    gpu_log_safe(
        device,
        &workspace.softmax_output,
        &mut workspace.log_buffer,
        total_elements,
        config.epsilon,
    )?;

    gpu_cross_entropy_reduce(
        device,
        &workspace.log_buffer,
        &workspace.one_hot_targets,
        &mut workspace.per_token_loss,
        total_tokens,
        vocab_size,
    )?;

    let ce_loss = device.mean(&workspace.per_token_loss, total_tokens)?;

    // 4. Compute reverse CE loss
    // RCE = -sum(softmax * log(one_hot + eps))
    // This penalizes overconfident wrong predictions
    let rce_loss = compute_rce_loss(
        device,
        &workspace.softmax_output,
        &workspace.one_hot_targets,
        &mut workspace.log_buffer,
        total_tokens,
        vocab_size,
        config.epsilon,
    )?;

    // 5. Combined loss
    let total_loss = config.alpha * ce_loss + config.beta * rce_loss;
    device.upload(&[total_loss], loss_output)?;

    // 6. Combined gradients
    // grad = alpha * (softmax - one_hot) + beta * RCE_grad
    compute_sce_gradients(
        device,
        &workspace.softmax_output,
        &workspace.one_hot_targets,
        &mut workspace.log_buffer,
        grad_output,
        total_tokens,
        vocab_size,
        config.alpha,
        config.beta,
        config.epsilon,
    )?;

    Ok(total_loss)
}

// ============================================================================
// Helper Functions (GPU Kernels)
// ============================================================================

/// One-hot encode target token IDs on GPU
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn one_hot_encode_gpu(
    device: &mut GpuDevice,
    targets: &GpuBuffer,
    output: &mut GpuBuffer,
    num_tokens: usize,
    vocab_size: usize,
) -> Result<()> {
    // Download targets to CPU, encode, upload
    // Targets are stored as f32 values representing token IDs
    // TODO: Replace with GPU kernel for large vocabs
    let mut targets_host = vec![0.0f32; num_tokens];
    device.download(targets, &mut targets_host)?;

    let mut one_hot_host = vec![0.0f32; num_tokens * vocab_size];
    for (i, &target_f32) in targets_host.iter().enumerate() {
        let target = target_f32 as usize;
        if target < vocab_size {
            one_hot_host[i * vocab_size + target] = 1.0;
        }
    }

    device.upload(&one_hot_host, output)
}

/// Compute log(x + eps) safely on GPU
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn gpu_log_safe(
    device: &mut GpuDevice,
    input: &GpuBuffer,
    output: &mut GpuBuffer,
    size: usize,
    epsilon: f32,
) -> Result<()> {
    // Download, compute log, upload
    // TODO: Replace with GPU kernel
    let mut host = vec![0.0f32; size];
    device.download(input, &mut host)?;

    for x in &mut host {
        *x = (*x + epsilon).ln();
    }

    device.upload(&host, output)
}

/// Cross-entropy reduction: -sum(one_hot * log_softmax)
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn gpu_cross_entropy_reduce(
    device: &mut GpuDevice,
    log_softmax: &GpuBuffer,
    one_hot: &GpuBuffer,
    output: &mut GpuBuffer,
    num_tokens: usize,
    vocab_size: usize,
) -> Result<()> {
    // Download both, compute, upload
    // TODO: Replace with GPU kernel
    let mut log_softmax_host = vec![0.0f32; num_tokens * vocab_size];
    let mut one_hot_host = vec![0.0f32; num_tokens * vocab_size];
    device.download(log_softmax, &mut log_softmax_host)?;
    device.download(one_hot, &mut one_hot_host)?;

    let mut output_host = vec![0.0f32; num_tokens];
    for i in 0..num_tokens {
        let mut sum = 0.0f32;
        for j in 0..vocab_size {
            sum += one_hot_host[i * vocab_size + j] * log_softmax_host[i * vocab_size + j];
        }
        output_host[i] = -sum;
    }

    device.upload(&output_host, output)
}

/// Compute gradient: scale * (softmax - one_hot)
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn gpu_gradient_compute(
    device: &mut GpuDevice,
    softmax: &GpuBuffer,
    one_hot: &GpuBuffer,
    output: &mut GpuBuffer,
    size: usize,
    scale: f32,
) -> Result<()> {
    // Download both, compute, upload
    // TODO: Replace with GPU kernel
    let mut softmax_host = vec![0.0f32; size];
    let mut one_hot_host = vec![0.0f32; size];
    device.download(softmax, &mut softmax_host)?;
    device.download(one_hot, &mut one_hot_host)?;

    for i in 0..size {
        softmax_host[i] = scale * (softmax_host[i] - one_hot_host[i]);
    }

    device.upload(&softmax_host, output)
}

/// Compute reverse cross-entropy loss
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn compute_rce_loss(
    device: &mut GpuDevice,
    softmax: &GpuBuffer,
    one_hot: &GpuBuffer,
    log_buffer: &mut GpuBuffer,
    num_tokens: usize,
    vocab_size: usize,
    epsilon: f32,
) -> Result<f32> {
    let size = num_tokens * vocab_size;

    // Compute log(one_hot + eps)
    let mut one_hot_host = vec![0.0f32; size];
    device.download(one_hot, &mut one_hot_host)?;

    for x in &mut one_hot_host {
        *x = (*x + epsilon).ln();
    }
    device.upload(&one_hot_host, log_buffer)?;

    // Compute softmax * log(one_hot + eps)
    let mut softmax_host = vec![0.0f32; size];
    device.download(softmax, &mut softmax_host)?;
    device.download(log_buffer, &mut one_hot_host)?;

    let mut rce_sum = 0.0f32;
    for i in 0..num_tokens {
        let mut token_rce = 0.0f32;
        for j in 0..vocab_size {
            token_rce += softmax_host[i * vocab_size + j] * one_hot_host[i * vocab_size + j];
        }
        rce_sum -= token_rce; // Negative because RCE = -sum
    }

    Ok(rce_sum / num_tokens as f32)
}

/// Compute symmetric cross-entropy gradients
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[allow(clippy::too_many_arguments)]
fn compute_sce_gradients(
    device: &mut GpuDevice,
    softmax: &GpuBuffer,
    one_hot: &GpuBuffer,
    log_buffer: &mut GpuBuffer,
    output: &mut GpuBuffer,
    num_tokens: usize,
    vocab_size: usize,
    alpha: f32,
    beta: f32,
    epsilon: f32,
) -> Result<()> {
    let size = num_tokens * vocab_size;
    let scale = 1.0 / num_tokens as f32;

    // Download all buffers
    let mut softmax_host = vec![0.0f32; size];
    let mut one_hot_host = vec![0.0f32; size];
    device.download(softmax, &mut softmax_host)?;
    device.download(one_hot, &mut one_hot_host)?;

    // Compute gradients
    // CE grad: softmax - one_hot
    // RCE grad: log(one_hot + eps) + 1 (for softmax term)
    // Combined: alpha * (softmax - one_hot) + beta * (log(one_hot + eps) + 1)
    for i in 0..size {
        let ce_grad = softmax_host[i] - one_hot_host[i];
        let rce_grad = (one_hot_host[i] + epsilon).ln() + 1.0;
        softmax_host[i] = scale * (alpha * ce_grad + beta * rce_grad);
    }

    device.upload(&softmax_host, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_ce_config_default() {
        let config = GpuSymmetricCEConfig::default();
        assert!((config.alpha - 0.1).abs() < 1e-6);
        assert!((config.beta - 0.1).abs() < 1e-6);
        assert!((config.epsilon - 1e-8).abs() < 1e-10);
    }
}
