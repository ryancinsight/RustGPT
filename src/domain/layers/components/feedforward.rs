//! Shared Feedforward Component
//!
//! This component provides a unified feedforward interface that can be used
//! by multiple architectures (Transformer, Diffusion, SSM).
//!
//! ## Memory Efficiency (Phase 5.1.1)
//!
//! The `forward_into()` method implements zero-allocation batch forwarding by:
//! 1. RichardsGlu: Reuses `batch_workspace` buffers for x1, x2, value, gate_sigma, gated
//! 2. MixtureOfExperts: Pre-allocates expert computation buffers (future optimization)
//! 3. Workspace reuse: Buffers are sized with power-of-2 to minimize reallocations

use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuComponent, GpuDevice};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::gpu_device_utils::resolve_or_create_gpu_device;
use crate::{
    common::errors::Result,
    domain::compute_backend::{ComputeBackend, resolve_compute_backend_strict_auto_gpu},
    domain::layers::components::{
        common::FeedForwardVariant, conditioning::apply_optional_delta_film,
    },
    domain::network::Layer,
};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::sync::{Arc, Mutex};

/// Shared feedforward component with workspace management
///
/// Consolidates feedforward variants (RichardsGlu, MixtureOfExperts) with
/// unified workspace allocation and reuse patterns for memory efficiency.
///
/// ## GPU Support (Phase 5.6)
///
/// Implements `GpuComponent` trait for unified GPU device management across
/// all shared components. Supports:
/// - Automatic GPU detection with strict no-fallback semantics
/// - GPU device attachment and management
/// - Capacity pre-allocation for zero-allocation reuse
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharedFeedforward {
    /// The underlying feedforward variant
    pub feedforward: FeedForwardVariant,

    /// Metadata for workspace capacity management
    /// Tracks last batch size and embedding dim to detect workspace resize needs
    #[serde(skip)]
    last_batch_size: Option<usize>,
    #[serde(skip)]
    last_embed_dim: Option<usize>,

    #[serde(skip, default)]
    compute_backend: ComputeBackend,

    /// GPU device for this component (Phase 5.6)
    /// If attached, enables GPU execution with strict no-fallback semantics
    #[serde(skip)]
    #[allow(dead_code)]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,
}

impl SharedFeedforward {
    /// Create a new shared feedforward component
    pub fn new(feedforward: FeedForwardVariant) -> Self {
        Self {
            feedforward,
            last_batch_size: None,
            last_embed_dim: None,
            compute_backend: ComputeBackend::Cpu,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            gpu_device: None,
        }
    }

    /// Forward pass through the feedforward network
    ///
    /// ## GPU Execution (Phase 5.6)
    ///
    /// When a GPU backend is selected and GPU device is attached, executes
    /// the feedforward computation on GPU with automatic buffer management.
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let (batch_size, embed_dim) = input.dim();
        self.last_batch_size = Some(batch_size);
        self.last_embed_dim = Some(embed_dim);

        // Strict GPU path: never fall back to CPU once a GPU backend is selected.
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            return self.forward_gpu(input).unwrap_or_else(|err| {
                panic!("SharedFeedforward GPU forward failed: {err}");
            });
        }

        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        if self.compute_backend.is_gpu() {
            panic!(
                "SharedFeedforward configured for GPU backend '{}' but this binary has no GPU features enabled.",
                self.compute_backend.as_str()
            );
        }

        // CPU path (default or when GPU not available)
        self.feedforward.forward(input)
    }

    /// Forward pass with in-place output buffer (Zero Allocation - Phase 5.1.1)
    ///
    /// Writes output directly to the provided buffer, eliminating intermediate
    /// allocations. This is the preferred method for batch processing where
    /// output buffers are pre-allocated from UnifiedLayerWorkspace.
    ///
    /// Workspace management:
    /// - RichardsGlu: Reuses batch_workspace buffers (x1, x2, value, gate_sigma, gated)
    /// - MixtureOfExperts: Workspace resized automatically via forward()
    /// - All buffers use power-of-2 sizing to minimize reallocations
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch_size, embed_dim)
    /// * `output` - Pre-allocated output buffer of shape (batch_size, embed_dim)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if output dimensions don't match input
    ///
    pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        let (batch_size, embed_dim) = input.dim();
        self.last_batch_size = Some(batch_size);
        self.last_embed_dim = Some(embed_dim);

        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            let gpu_out = self.forward_gpu(input)?;
            if gpu_out.dim() != output.dim() {
                return Err(crate::common::errors::ModelError::InvalidInput {
                    message: format!(
                        "Output dimension mismatch: expected {:?}, got {:?}",
                        gpu_out.dim(),
                        output.dim()
                    ),
                });
            }
            output.assign(&gpu_out);
            return Ok(());
        }

        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        if self.compute_backend.is_gpu() {
            return Err(crate::common::errors::ModelError::Backend {
                message: format!(
                    "SharedFeedforward configured for GPU backend '{}' but this binary has no GPU features enabled.",
                    self.compute_backend.as_str()
                ),
            });
        }
        self.feedforward.forward_into(input, output)
    }

    /// Clear internal caches while preserving workspace allocations
    ///
    /// This is useful after inference steps to release cached inputs/gradients
    /// while keeping the pre-allocated workspace buffers ready for reuse.
    pub fn clear_cache(&mut self) {
        // Variants manage their own caches; this is a hook for future optimization
        // Could be extended to clear cached_input, etc. in RichardsGlu/MoE
    }

    /// Get workspace statistics for memory monitoring
    ///
    /// Returns `(last_batch_size, last_embed_dim, workspace_info)`
    pub fn workspace_info(&self) -> (Option<usize>, Option<usize>) {
        (self.last_batch_size, self.last_embed_dim)
    }

    /// Set runtime compute backend.
    #[inline]
    pub fn set_compute_backend(&mut self, compute_backend: ComputeBackend) {
        self.set_compute_backend_checked(compute_backend)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to set SharedFeedforward backend '{}': {}",
                    compute_backend.as_str(),
                    err
                )
            });
    }

    /// Set runtime compute backend with strict validation.
    ///
    /// When a GPU backend is selected, this creates the GPU device and attaches it.
    pub fn set_compute_backend_checked(&mut self, compute_backend: ComputeBackend) -> Result<()> {
        self.compute_backend = compute_backend;

        if compute_backend.is_gpu() {
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            {
                let (device_arc, _) = resolve_or_create_gpu_device(
                    self.gpu_device.clone(),
                    compute_backend,
                    "feedforward",
                )?;
                self.gpu_device = Some(device_arc.clone());
                self.feedforward.set_gpu_device(device_arc);

                return Ok(());
            }

            #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
            {
                return Err(crate::common::errors::ModelError::Backend {
                    message: format!(
                        "SharedFeedforward requested GPU backend '{}' but this binary was built without GPU features.",
                        compute_backend.as_str()
                    ),
                });
            }
        }

        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            self.gpu_device = None;
        }

        Ok(())
    }

    /// Resolve and apply `AutoGpu` backend preference.
    ///
    /// This uses strict runtime detection: GPU is preferred, CPU is used only when
    /// no GPU is detected, and feature mismatches return an error.
    pub fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let backend = resolve_compute_backend_strict_auto_gpu()?;
        self.set_compute_backend_checked(backend)
    }

    /// Set GPU device for execution.
    #[inline]
    pub fn set_gpu_device(
        &mut self,
        device: std::sync::Arc<std::sync::Mutex<crate::domain::compute::GpuDevice>>,
    ) {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            self.gpu_device = Some(device.clone());
        }
        self.feedforward.set_gpu_device(device);
    }

    /// Get runtime compute backend.
    #[inline]
    pub fn compute_backend(&self) -> ComputeBackend {
        self.compute_backend
    }

    /// Forward pass on GPU with automatic kernel fusion
    ///
    /// Uses optimized GPU kernels for feedforward computation:
    ///
    /// **RichardsGlu**: Fused kernel implementation
    /// - Pass 1: GEMM for x1, x2 + Richards activation + gating
    /// - Pass 2: GEMM for output projection (w_out)
    /// - Single upload/download cycle for maximum efficiency
    ///
    /// **MixtureOfExperts**: GPU-accelerated router and expert dispatch
    /// - Step 1: Router GEMM `input @ W_router` → routing_logits
    /// - Step 2: Softmax (masked for top-k) → routing probabilities
    /// - Step 3: Top-k selection and expert routing
    /// - Step 4: Parallel expert GEMMs on GPU
    /// - Step 5: Weighted combination of expert outputs
    /// - Single upload/download cycle for maximum efficiency
    ///
    /// Returns error if GPU features are not enabled or GPU is unavailable.
    pub fn forward_gpu(&mut self, _input: &Array2<f32>) -> Result<Array2<f32>> {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            self.feedforward.ensure_gpu_device_auto_detect()?;
            self.feedforward.forward_gpu(_input)
        }
        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        {
            Err(crate::common::errors::ModelError::Backend {
                message: "GPU features not enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal".to_string(),
            })
        }
    }

    /// Backward pass through the feedforward network
    pub fn backward(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            return self
                .compute_gradients_gpu(input, output_grads)
                .unwrap_or_else(|err| panic!("SharedFeedforward GPU backward failed: {err}"));
        }

        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        if self.compute_backend.is_gpu() {
            panic!(
                "SharedFeedforward configured for GPU backend '{}' but this binary has no GPU features enabled.",
                self.compute_backend.as_str()
            );
        }

        self.feedforward.compute_gradients(input, output_grads)
    }

    /// Backward pass on GPU-capable variants.
    ///
    /// Returns `(input_grads, param_grads)` to keep compatibility with the shared
    /// gradient-routing path used by Transformer/Diffusion blocks.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn compute_gradients_gpu(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        self.feedforward.compute_gradients_gpu(input, output_grads)
    }

    #[inline]
    pub fn variant(&self) -> &FeedForwardVariant {
        &self.feedforward
    }

    #[inline]
    pub fn variant_mut(&mut self) -> &mut FeedForwardVariant {
        &mut self.feedforward
    }

    #[inline]
    pub fn as_moe(&self) -> Option<&crate::domain::mixtures::moe::MixtureOfExperts> {
        self.feedforward.as_moe()
    }

    #[inline]
    pub fn as_moe_mut(&mut self) -> Option<&mut crate::domain::mixtures::moe::MixtureOfExperts> {
        self.feedforward.as_moe_mut()
    }

    /// Apply gradients to the feedforward network
    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        self.feedforward.apply_gradients(param_grads, lr)
    }

    /// Get the number of parameters
    pub fn parameters(&self) -> usize {
        self.feedforward.parameters()
    }

    /// Get the weight norm
    pub fn weight_norm(&self) -> f32 {
        self.feedforward.weight_norm()
    }

    /// Zero out gradients
    pub fn zero_gradients(&mut self) {
        self.feedforward.zero_gradients()
    }

    /// Get the layer type name
    pub fn layer_type(&self) -> &str {
        match &self.feedforward {
            FeedForwardVariant::RichardsGlu(_) => "RichardsGlu",
            FeedForwardVariant::MixtureOfExperts(_) => "MixtureOfExperts",
        }
    }

    pub fn forward_with_token_head_activity(
        &mut self,
        input: &Array2<f32>,
        head_activity_ratio: Option<f32>,
        head_activity_vec: Option<&[f32]>,
        token_head_activity_vec: Option<&[f32]>,
    ) -> Array2<f32> {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            self.feedforward
                .ensure_gpu_device_auto_detect()
                .unwrap_or_else(|err| {
                    panic!(
                        "SharedFeedforward failed to attach GPU device for activity-aware forward: {err}"
                    )
                });

            if matches!(self.feedforward, FeedForwardVariant::RichardsGlu(_)) {
                return self.forward_gpu(input).unwrap_or_else(|err| {
                    panic!("SharedFeedforward RichardsGlu GPU forward failed: {err}");
                });
            }

            if let FeedForwardVariant::MixtureOfExperts(layer) = &mut self.feedforward {
                return layer.forward_with_head_features_and_token_activity(
                    input,
                    head_activity_ratio,
                    head_activity_vec,
                    token_head_activity_vec,
                );
            }
        }

        // CPU path
        self.feedforward.forward_with_token_head_activity(
            input,
            head_activity_ratio,
            head_activity_vec,
            token_head_activity_vec,
        )
    }

    /// In-place feedforward with optional token/head activity features.
    ///
    /// Uses true in-place execution for RichardsGlu; MoE currently falls back to assignment
    /// from the activity-aware forward path when head-conditioning is required.
    pub fn forward_with_token_head_activity_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
        head_activity_ratio: Option<f32>,
        head_activity_vec: Option<&[f32]>,
        token_head_activity_vec: Option<&[f32]>,
    ) -> Result<()> {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            self.feedforward.ensure_gpu_device_auto_detect()?;

            if matches!(self.feedforward, FeedForwardVariant::RichardsGlu(_)) {
                let gpu_out = self.forward_gpu(input)?;
                if gpu_out.dim() != output.dim() {
                    return Err(crate::common::errors::ModelError::InvalidInput {
                        message: format!(
                            "Output dimension mismatch: expected {:?}, got {:?}",
                            gpu_out.dim(),
                            output.dim()
                        ),
                    });
                }
                output.assign(&gpu_out);
                return Ok(());
            }

            if let FeedForwardVariant::MixtureOfExperts(layer) = &mut self.feedforward {
                let out = layer.forward_with_head_features_and_token_activity(
                    input,
                    head_activity_ratio,
                    head_activity_vec,
                    token_head_activity_vec,
                );
                if out.dim() != output.dim() {
                    return Err(crate::common::errors::ModelError::InvalidInput {
                        message: format!(
                            "Output dimension mismatch: expected {:?}, got {:?}",
                            out.dim(),
                            output.dim()
                        ),
                    });
                }
                output.assign(&out);
                return Ok(());
            }
        }

        // CPU path
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => layer.forward_into(input, output),
            FeedForwardVariant::MixtureOfExperts(layer) => {
                let needs_activity_path = head_activity_ratio.is_some()
                    || head_activity_vec.is_some()
                    || token_head_activity_vec.is_some();
                if needs_activity_path {
                    let out = layer.forward_with_head_features_and_token_activity(
                        input,
                        head_activity_ratio,
                        head_activity_vec,
                        token_head_activity_vec,
                    );
                    if out.dim() != output.dim() {
                        return Err(crate::common::errors::ModelError::InvalidInput {
                            message: format!(
                                "Output dimension mismatch: expected {:?}, got {:?}",
                                out.dim(),
                                output.dim()
                            ),
                        });
                    }
                    output.assign(&out);
                    Ok(())
                } else {
                    layer.forward_into(input, output)
                }
            }
        }
    }

    /// Forward pass with FiLM conditioning and optional token head activity
    pub fn forward_with_film(
        &mut self,
        input: &Array2<f32>,
        gamma: Option<&Array1<f32>>,
        beta: Option<&Array1<f32>>,
        head_activity_ratio: Option<f32>,
        head_activity_vec: Option<&[f32]>,
        token_head_activity_vec: Option<&[f32]>,
    ) -> Array2<f32> {
        // FiLM conditioning is CPU-only for now
        let conditioned =
            apply_optional_delta_film(input, gamma.map(|g| g.view()), beta.map(|b| b.view()));
        self.forward_with_token_head_activity(
            conditioned.as_ref(),
            head_activity_ratio,
            head_activity_vec,
            token_head_activity_vec,
        )
    }

    pub fn forward_step_into(
        &mut self,
        input: &ArrayView1<f32>,
        output: &mut Array1<f32>,
        head_activity_ratio: Option<f32>,
        head_activity_vec: Option<&[f32]>,
        token_head_activity: Option<f32>,
    ) {
        // Streaming step mode uses CPU path
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => {
                layer.forward_step_into(input, output);
            }
            FeedForwardVariant::MixtureOfExperts(layer) => {
                layer.forward_step_with_head_features_into(
                    input,
                    output,
                    head_activity_ratio,
                    head_activity_vec,
                    token_head_activity,
                );
            }
        }
    }

    pub fn set_training_progress(&mut self, progress: f64) {
        // Training progress is CPU-only
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => layer.set_training_progress(progress),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.set_training_progress(progress),
        }
    }

    // --- GPU Support (Phase 5.3) ---

    /// GPU-accelerated feedforward forward pass.
    ///
    /// When a GPU backend is selected and available, this method computes the
    /// feedforward transformation on GPU hardware. Requires strict no-fallback
    /// mode—will return an error if GPU operations are unavailable.
    ///
    /// # Arguments
    /// * `gpu_input` - Input buffer on GPU (batch_size * embed_dim f32 elements)
    /// * `gpu_output` - Output buffer on GPU (batch_size * embed_dim f32 elements)
    /// * `gpu_device` - GPU device context with kernels
    /// * `batch_size` - Number of samples in the batch
    /// * `embed_dim` - Embedding dimension
    ///
    /// # Returns
    /// Ok(()) on success, or an error if GPU operation fails
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_gpu_buffer(
        &mut self,
        gpu_input: &crate::domain::compute::GpuBuffer,
        gpu_output: &mut crate::domain::compute::GpuBuffer,
        gpu_device: &mut crate::domain::compute::GpuDevice,
        batch_size: usize,
        embed_dim: usize,
    ) -> crate::common::errors::Result<()> {
        // Verify GPU backend is selected
        if !self.compute_backend.is_gpu() {
            return Err(crate::common::errors::ModelError::Backend {
                message: format!(
                    "SharedFeedforward::forward_gpu called with non-GPU backend '{}'. \
                     Use forward() for CPU computation.",
                    self.compute_backend.as_str()
                ),
            });
        }

        self.feedforward.ensure_gpu_device_auto_detect()?;

        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => {
                if embed_dim != layer.w1.nrows() || embed_dim != layer.w_out.ncols() {
                    return Err(crate::common::errors::ModelError::InvalidInput {
                        message: format!(
                            "RichardsGlu GPU buffer forward dimension mismatch: input embed_dim={}, w1 rows={}, w_out cols={}",
                            embed_dim,
                            layer.w1.nrows(),
                            layer.w_out.ncols()
                        ),
                    });
                }

                // True on-device execution: no host roundtrip for input/output tensors.
                let (pool, ops) = gpu_device.execution_context();
                let (x1, x2, value, gate_sigma, gated) =
                    layer.forward_gpu_kernel(pool, ops, gpu_input, gpu_output, batch_size)?;

                // This buffer path is forward-only; release temporary intermediates.
                pool.deallocate(x1);
                pool.deallocate(x2);
                pool.deallocate(value);
                pool.deallocate(gate_sigma);
                pool.deallocate(gated);
                Ok(())
            }
            FeedForwardVariant::MixtureOfExperts(layer) => {
                let (pool, ops) = gpu_device.execution_context();
                let output = layer.forward_gpu_kernel(pool, ops, gpu_input, batch_size, embed_dim, embed_dim)?;
                ops.copy_within_device(pool, &output, gpu_output, batch_size * embed_dim)?;
                pool.deallocate(output);
                Ok(())
            }
        }
    }

    /// Check if GPU execution is available for this feedforward component.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn is_gpu_available(&self) -> bool {
        self.compute_backend.is_gpu()
    }
}

// ============================================================================
// GPU Component Trait Implementation (Phase 5.6)
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuComponent for SharedFeedforward {
    /// Attach a pre-configured GPU device
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.gpu_device = Some(device.clone());
        // Also set device on the underlying feedforward variant
        self.feedforward.set_gpu_device(device);
    }

    /// Enable GPU with automatic detection (strict no-fallback)
    ///
    /// # Errors
    /// Returns an error if:
    /// - No GPU is detected on the system
    /// - GPU feature flags are not enabled
    /// - GPU initialization fails
    fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;
        self.set_compute_backend_checked(device.backend())
    }

    /// Check if GPU is ready for execution
    fn is_gpu_ready(&self) -> bool {
        self.gpu_device.is_some() && self.compute_backend.is_gpu()
    }

    /// Get the GPU backend name if attached
    fn gpu_backend_name(&self) -> Option<&'static str> {
        if let Some(device_arc) = &self.gpu_device {
            if let Ok(device) = device_arc.lock() {
                return Some(device.backend().as_str());
            }
        }
        None
    }

    /// Ensure buffers have sufficient capacity
    fn ensure_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        _seq_len: usize,
    ) -> Result<()> {
        // For feedforward, we mainly need to track dimensions for FFN sizing
        self.last_batch_size = Some(batch_size);
        self.last_embed_dim = Some(embed_dim);

        // Ensure GPU device has this capacity available
        if let Some(device_arc) = &self.gpu_device {
            let mut device =
                device_arc
                    .lock()
                    .map_err(|_| crate::common::errors::ModelError::Backend {
                        message: "Failed to acquire GPU device lock".to_string(),
                    })?;

            // Allocate input buffer if needed
            let input_size = batch_size * embed_dim;
            let _ = device.allocate_f32(input_size);

            // Allocate output buffer if needed
            let _ = device.allocate_f32(input_size);
        }

        Ok(())
    }

    /// Get the attached GPU device for direct operations
    fn gpu_device(&self) -> Option<Arc<Mutex<GpuDevice>>> {
        self.gpu_device.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::SharedFeedforward;
    use crate::{
        domain::compute_backend::ComputeBackend,
        domain::layers::components::common::FeedForwardVariant,
        domain::mixtures::{
            gating::GatingConfig,
            moe::{ExpertRouterConfig, LearnedKAdapter, MixtureOfExperts},
        },
    };
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    use std::sync::{Arc, Mutex};

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    use crate::domain::{
        compute::{GpuComponent, GpuDevice},
        compute_backend::resolve_compute_backend_strict_auto_gpu,
    };

    #[test]
    fn test_shared_feedforward_forwards_token_head_activity_to_moe() {
        let mut config = ExpertRouterConfig {
            num_experts: 4,
            expert_hidden_dim: 16,
            diversity_weight: 0.005,
            gating: GatingConfig {
                num_active: 3,
                load_balance_weight: 0.01,
                sparsity_weight: 0.001,
                ..Default::default()
            },
            ..Default::default()
        };
        config.use_head_conditioning = true;
        config.use_learned_k_adaptation = true;

        let mut moe = MixtureOfExperts::new(32, 8, config);
        moe.k_adapter = Some(LearnedKAdapter {
            w: ndarray::Array2::from_shape_vec((2, 1), vec![0.0, 20.0]).unwrap(),
            b: ndarray::Array2::from_shape_vec((1, 1), vec![-10.0]).unwrap(),
        });

        let mut processor =
            SharedFeedforward::new(FeedForwardVariant::MixtureOfExperts(Box::new(moe)));

        let input = ndarray::Array2::<f32>::from_shape_vec((2, 32), vec![0.1; 64]).unwrap();
        let token_h = vec![0.0f32, 1.0f32];

        let _out = processor.forward_with_token_head_activity(
            &input,
            Some(0.0),
            None,
            Some(token_h.as_slice()),
        );

        let FeedForwardVariant::MixtureOfExperts(moe) = &processor.feedforward else {
            panic!("expected MoE feedforward");
        };

        let router_in = moe.test_cached_router_input().unwrap();
        assert!((router_in[[0, 32]] - 0.0).abs() < 1e-6);
        assert!((router_in[[1, 32]] - 1.0).abs() < 1e-6);

        let alpha = moe.test_cached_k_alpha().unwrap();
        assert!(alpha[0] < 0.01);
        assert!(alpha[1] > 0.99);
    }

    #[test]
    fn test_shared_feedforward_zero_allocation_forward_into() {
        // Test RichardsGlu forward_into (zero-allocation)
        let richards_glu = crate::domain::richards::RichardsGlu::new(8, 16);
        let mut processor =
            SharedFeedforward::new(FeedForwardVariant::RichardsGlu(Box::new(richards_glu)));

        let input = ndarray::Array2::from_elem((2, 8), 0.5);
        let mut output = ndarray::Array2::zeros((2, 8));

        processor.forward_into(&input, &mut output).unwrap();

        // Verify output is populated
        assert!(
            !output.iter().all(|&x| x == 0.0),
            "output should be non-zero"
        );

        // Compare with regular forward
        let output_regular = processor.forward(&input);
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                // Should be very close (may differ slightly due to numerical precision)
                assert!(
                    (output[[i, j]] - output_regular[[i, j]]).abs() < 1e-5,
                    "forward_into and forward results differ at [{}, {}]",
                    i,
                    j
                );
            }
        }

        // Verify workspace info is tracked
        let (batch_size, embed_dim) = processor.workspace_info();
        assert_eq!(batch_size, Some(2));
        assert_eq!(embed_dim, Some(8));
    }

    #[test]
    fn test_shared_feedforward_set_compute_backend_checked_cpu() {
        let richards_glu = crate::domain::richards::RichardsGlu::new(8, 16);
        let mut processor =
            SharedFeedforward::new(FeedForwardVariant::RichardsGlu(Box::new(richards_glu)));

        processor
            .set_compute_backend_checked(ComputeBackend::Cpu)
            .expect("CPU backend should always be accepted");

        assert_eq!(processor.compute_backend(), ComputeBackend::Cpu);
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_shared_feedforward_reuses_preattached_gpu_device() {
        let backend = match resolve_compute_backend_strict_auto_gpu() {
            Ok(backend) => backend,
            Err(_) => return,
        };

        let richards_glu = crate::domain::richards::RichardsGlu::new(8, 16);
        let mut processor =
            SharedFeedforward::new(FeedForwardVariant::RichardsGlu(Box::new(richards_glu)));

        let device = Arc::new(Mutex::new(
            GpuDevice::new(backend).expect("resolved backend should create a GPU device"),
        ));
        processor.set_gpu_device(device.clone());

        processor
            .set_compute_backend_checked(backend)
            .expect("pre-attached matching GPU device should be reused");

        assert_eq!(processor.compute_backend(), backend);

        let attached = processor
            .gpu_device()
            .expect("GPU device should remain attached after backend setup");
        assert!(Arc::ptr_eq(&attached, &device));
    }
}
