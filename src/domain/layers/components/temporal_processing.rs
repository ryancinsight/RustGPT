//! Shared Temporal Processing Component
//!
//! This component provides a unified interface for temporal processing
//! (attention, RG-LRU, Mamba) that can be used by multiple architectures.

use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};

use crate::{
    common::errors::Result,
    domain::{
        attention::poly_attention::{DegreeAdaptationMetrics, PolyAttention},
        compute::GpuDevice,
        compute_backend::{ComputeBackend, resolve_compute_backend_strict_auto_gpu},
        layers::components::{
            common::{TemporalMixingLayer, TitanMemoryWorkspace},
            conditioning::apply_optional_delta_film,
        },
        models::config::TitanMemoryConfig,
        network::Layer,
    },
};
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::GpuComponent;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::gpu_device_utils::resolve_or_create_gpu_device;

/// Shared temporal processing component
///
/// ## GPU Support (Phase 5.6)
///
/// Implements `GpuComponent` trait for unified GPU device management.
/// Supports automatic GPU detection with strict no-fallback semantics.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharedTemporalProcessing {
    /// The underlying temporal mixing layer
    pub temporal_mixing: TemporalMixingLayer,
    /// Window size for attention-based mixing
    pub window_size: Option<usize>,
    /// Use adaptive window sizing
    pub use_adaptive_window: bool,

    /// Runtime compute backend.
    #[serde(skip, default)]
    compute_backend: ComputeBackend,

    /// GPU device for this component (Phase 5.6)
    /// If attached, enables GPU execution with strict no-fallback semantics
    #[serde(skip)]
    #[allow(dead_code)]
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,
}

/// Consolidated head-activity views from temporal mixing.
#[derive(Clone, Copy, Debug)]
pub struct HeadActivitySummary<'a> {
    pub ratio: f32,
    pub head_activity: Option<&'a [f32]>,
    pub token_head_activity: Option<&'a [f32]>,
}

/// Consolidated MoH telemetry collected from a temporal mixer.
#[derive(Debug, Default)]
pub struct TemporalMoeMetrics {
    pub tau_metrics: Option<(f32, f32)>,
    pub pred_norm: Option<f32>,
    pub per_head_metrics: Vec<(f32, usize)>,
}

impl SharedTemporalProcessing {
    /// Create a new shared temporal processing component
    pub fn new(
        temporal_mixing: TemporalMixingLayer,
        window_size: Option<usize>,
        use_adaptive_window: bool,
    ) -> Self {
        Self {
            temporal_mixing,
            window_size,
            use_adaptive_window,
            compute_backend: ComputeBackend::Cpu,
            gpu_device: None,
        }
    }

    /// Forward pass through the temporal processing layer
    ///
    /// Uses the Layer trait for zero-cost abstraction, eliminating
    /// redundant match statements across all temporal mixing variants.
    ///
    /// ## GPU Execution (Phase 5.6)
    ///
    /// When a GPU backend is selected and a GPU device is attached, this method
    /// executes the temporal mixing on GPU with automatic buffer management.
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Strict GPU path: never fall back to CPU once a GPU backend is selected.
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            return self.forward_gpu(input).unwrap_or_else(|err| {
                panic!("SharedTemporalProcessing GPU forward failed: {err}");
            });
        }

        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        if self.compute_backend.is_gpu() {
            panic!(
                "SharedTemporalProcessing configured for GPU backend '{}' but this binary has no GPU features enabled.",
                self.compute_backend.as_str()
            );
        }

        // CPU path
        self.prepare_forward();
        self.temporal_mixing.forward(input)
    }

    /// Forward pass with explicit causal masking control
    pub fn forward_with_causal(&mut self, input: &Array2<f32>, causal: bool) -> Array2<f32> {
        // Strict GPU path: apply causal flag then dispatch to GPU.
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            if let TemporalMixingLayer::Attention(attn) = &mut self.temporal_mixing {
                attn.set_last_causal(causal);
            }
            return self.forward_gpu(input).unwrap_or_else(|err| {
                panic!("SharedTemporalProcessing GPU forward_with_causal failed: {err}");
            });
        }

        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        if self.compute_backend.is_gpu() {
            panic!(
                "SharedTemporalProcessing configured for GPU backend '{}' but this binary has no GPU features enabled.",
                self.compute_backend.as_str()
            );
        }

        // CPU path
        self.prepare_forward();
        self.temporal_mixing.forward_with_causal(input, causal)
    }

    /// Forward pass with FiLM conditioning and explicit causal masking
    pub fn forward_with_film(
        &mut self,
        input: &Array2<f32>,
        gamma: Option<&Array1<f32>>,
        beta: Option<&Array1<f32>>,
        causal: bool,
    ) -> Array2<f32> {
        if gamma.is_none() && beta.is_none() {
            return self.forward_with_causal(input, causal);
        }
        let conditioned =
            apply_optional_delta_film(input, gamma.map(|g| g.view()), beta.map(|b| b.view()));
        self.forward_with_causal(conditioned.as_ref(), causal)
    }

    /// Forward pass with in-place output buffer (Zero Allocation)
    ///
    /// Writes output directly to the provided buffer, eliminating intermediate
    /// allocations. This is the preferred method for batch processing where
    /// output buffers are pre-allocated from UnifiedLayerWorkspace.
    pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            let gpu_out = self.forward_gpu(input)?;
            if output.raw_dim() == gpu_out.raw_dim() {
                output.assign(&gpu_out);
            } else {
                *output = gpu_out;
            }
            return Ok(());
        }

        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        if self.compute_backend.is_gpu() {
            return Err(crate::common::errors::ModelError::Backend {
                message: format!(
                    "SharedTemporalProcessing configured for GPU backend '{}' but this binary has no GPU features enabled.",
                    self.compute_backend.as_str()
                ),
            });
        }

        self.prepare_forward();
        self.temporal_mixing.forward_into(input, output)
    }

    /// Forward pass with causal control and in-place output (Zero Allocation)
    ///
    /// Writes output directly to the provided buffer with explicit causal masking control.
    pub fn forward_with_causal_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
        causal: bool,
    ) -> Result<()> {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            if let TemporalMixingLayer::Attention(attn) = &mut self.temporal_mixing {
                attn.set_last_causal(causal);
            }
            let gpu_out = self.forward_gpu(input)?;
            if output.raw_dim() == gpu_out.raw_dim() {
                output.assign(&gpu_out);
            } else {
                *output = gpu_out;
            }
            return Ok(());
        }

        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        if self.compute_backend.is_gpu() {
            return Err(crate::common::errors::ModelError::Backend {
                message: format!(
                    "SharedTemporalProcessing configured for GPU backend '{}' but this binary has no GPU features enabled.",
                    self.compute_backend.as_str()
                ),
            });
        }

        // CPU path
        self.prepare_forward();
        self.temporal_mixing
            .forward_with_causal_into(input, output, causal)
    }

    /// Returns true when Titan memory should be fused externally to the temporal output.
    ///
    /// Attention and Titans variants already encapsulate memory/context paths.
    /// SSM/recurrent variants (RG-LRU/Mamba/Mamba2) benefit from explicit Titan fusion.
    #[inline]
    pub fn needs_external_titan_memory(&self) -> bool {
        !matches!(
            self.temporal_mixing,
            TemporalMixingLayer::Attention(_) | TemporalMixingLayer::Titans(_)
        )
    }

    /// Forward with optional Titan memory fusion for non-attention temporal mixers.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn forward_with_titan_fusion(
        &mut self,
        input: &Array2<f32>,
        causal: bool,
        titan_memory: &TitanMemoryConfig,
        titan_workspace: &mut TitanMemoryWorkspace,
    ) -> Array2<f32> {
        let mut out = self.forward_with_causal(input, causal);
        if self.needs_external_titan_memory() {
            titan_memory.apply_into_out_with_workspace(&mut out, input, titan_workspace);
        }
        out
    }

    /// In-place forward with optional Titan memory fusion for non-attention mixers.
    #[inline]
    pub(crate) fn forward_with_titan_fusion_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
        causal: bool,
        titan_memory: &TitanMemoryConfig,
        titan_workspace: &mut TitanMemoryWorkspace,
    ) -> Result<()> {
        self.forward_with_causal_into(input, output, causal)?;
        if self.needs_external_titan_memory() {
            titan_memory.apply_into_out_with_workspace(output, input, titan_workspace);
        }
        Ok(())
    }

    /// Forward with default temporal-mixer behavior plus Titan fusion for SSM variants.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn forward_with_titan_fusion_default(
        &mut self,
        input: &Array2<f32>,
        titan_memory: &TitanMemoryConfig,
        titan_workspace: &mut TitanMemoryWorkspace,
    ) -> Array2<f32> {
        let mut out = self.forward(input);
        if self.needs_external_titan_memory() {
            titan_memory.apply_into_out_with_workspace(&mut out, input, titan_workspace);
        }
        out
    }

    /// In-place default forward with Titan fusion for SSM/recurrent variants.
    #[inline]
    pub(crate) fn forward_with_titan_fusion_default_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
        titan_memory: &TitanMemoryConfig,
        titan_workspace: &mut TitanMemoryWorkspace,
    ) -> Result<()> {
        self.forward_into(input, output)?;
        if self.needs_external_titan_memory() {
            titan_memory.apply_into_out_with_workspace(output, input, titan_workspace);
        }
        Ok(())
    }

    /// Step-mode Titan fusion for streaming paths.
    #[inline]
    pub(crate) fn fuse_titan_step_into(
        &self,
        input: &ArrayView1<'_, f32>,
        output: &mut Array1<f32>,
        titan_memory: &TitanMemoryConfig,
        titan_workspace: &mut TitanMemoryWorkspace,
    ) {
        if self.needs_external_titan_memory() {
            titan_memory.apply_step_into(input, output, titan_workspace);
        }
    }

    /// Prepares the layer for a forward pass (e.g. setting window size)
    fn prepare_forward(&mut self) {
        // Set window size if using adaptive window and it's attention-based
        if self.use_adaptive_window {
            if let TemporalMixingLayer::Attention(attn) = &mut self.temporal_mixing {
                if let Some(window_size) = self.window_size {
                    attn.set_window_size(Some(window_size));
                }
            }
        }
    }

    /// Backward pass through the temporal processing layer
    ///
    /// Uses compute_gradients from Layer trait for consistent
    /// gradient computation across all temporal mixing variants.
    pub fn backward(
        &mut self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        self.compute_gradients(input, output_grads)
    }

    /// Apply gradients to the temporal processing layer
    ///
    /// Uses Layer trait method for zero-cost delegation.
    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        self.temporal_mixing.apply_gradients(param_grads, lr)
    }

    /// Get the number of parameters
    ///
    /// Uses Layer trait method for zero-cost delegation.
    pub fn parameters(&self) -> usize {
        self.temporal_mixing.parameters()
    }

    /// Get the weight norm
    ///
    /// Uses Layer trait method for zero-cost delegation.
    pub fn weight_norm(&self) -> f32 {
        self.temporal_mixing.weight_norm()
    }

    /// Zero out gradients
    ///
    /// Uses Layer trait method for zero-cost delegation.
    pub fn zero_gradients(&mut self) {
        self.temporal_mixing.zero_gradients()
    }

    /// Set training progress
    ///
    /// Uses Layer trait method for zero-cost delegation.
    pub fn set_training_progress(&mut self, progress: f64) {
        self.temporal_mixing.set_training_progress(progress);
    }

    /// Get the layer type name
    pub fn layer_type(&self) -> &'static str {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(_) => "Attention",
            TemporalMixingLayer::RgLru(_) => "RG-LRU",
            TemporalMixingLayer::Mamba(_) => "Mamba",
            TemporalMixingLayer::Mamba2(_) => "Mamba2",
            TemporalMixingLayer::RgLruMoH(_) => "RG-LRU-MoH",
            TemporalMixingLayer::MambaMoH(_) => "Mamba-MoH",
            TemporalMixingLayer::Mamba2MoH(_) => "Mamba2-MoH",
            TemporalMixingLayer::Titans(_) => "TitansMAC",
        }
    }

    /// Set window size for attention-based temporal mixing
    pub fn set_window_size(&mut self, window_size: Option<usize>) {
        self.window_size = window_size;
        if let TemporalMixingLayer::Attention(layer) = &mut self.temporal_mixing {
            layer.set_window_size(window_size);
        }
    }

    /// Returns true if the temporal mixer uses the shared GPU device (attention).
    #[inline]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn uses_shared_gpu_device(&self) -> bool {
        matches!(self.temporal_mixing, TemporalMixingLayer::Attention(_))
    }

    /// Returns true if the temporal mixer uses a variant-local GPU backend (SSM variants).
    #[inline]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn uses_variant_local_gpu_backend(&self) -> bool {
        matches!(
            self.temporal_mixing,
            TemporalMixingLayer::RgLru(_)
                | TemporalMixingLayer::RgLruMoH(_)
                | TemporalMixingLayer::Mamba(_)
                | TemporalMixingLayer::Mamba2(_)
                | TemporalMixingLayer::MambaMoH(_)
                | TemporalMixingLayer::Mamba2MoH(_)
        )
    }

    /// Set runtime compute backend.
    pub fn set_compute_backend(&mut self, compute_backend: ComputeBackend) {
        self.set_compute_backend_checked(compute_backend)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to set SharedTemporalProcessing backend '{}': {}",
                    compute_backend.as_str(),
                    err
                )
            });
    }

    /// Set runtime compute backend with strict validation.
    ///
    /// When a GPU backend is selected, this eagerly validates GPU availability.
    /// Attention variants attach a shared GPU device, while SSM variants rely on
    /// their own strict checked backend setup and internal caches.
    pub fn set_compute_backend_checked(&mut self, compute_backend: ComputeBackend) -> Result<()> {
        if compute_backend.is_gpu() {
            match &self.temporal_mixing {
                TemporalMixingLayer::Titans(_) => {
                    return Err(crate::common::errors::ModelError::Backend {
                        message: format!(
                            "Temporal mixer '{}' does not have GPU kernels yet. No CPU fallback is allowed.",
                            self.layer_type()
                        ),
                    });
                }
                _ => {}
            }

            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            {
                if self.uses_shared_gpu_device() {
                    let (device_arc, _) = resolve_or_create_gpu_device(
                        self.gpu_device.clone(),
                        compute_backend,
                        "temporal",
                    )?;

                    self.gpu_device = Some(device_arc.clone());
                    self.temporal_mixing.set_gpu_device(device_arc);
                    self.temporal_mixing.set_compute_backend(compute_backend);
                } else {
                    self.gpu_device = None;
                    match &mut self.temporal_mixing {
                        TemporalMixingLayer::RgLru(layer) => {
                            layer.set_compute_backend_checked(compute_backend)?;
                        }
                        TemporalMixingLayer::RgLruMoH(layer) => {
                            layer.set_compute_backend_checked(compute_backend)?;
                        }
                        TemporalMixingLayer::Mamba(layer) => {
                            layer.set_compute_backend_checked(compute_backend)?;
                        }
                        TemporalMixingLayer::Mamba2(layer) => {
                            layer.set_compute_backend_checked(compute_backend)?;
                        }
                        TemporalMixingLayer::MambaMoH(layer) => {
                            layer.set_compute_backend_checked(compute_backend)?;
                        }
                        TemporalMixingLayer::Mamba2MoH(layer) => {
                            layer.set_compute_backend_checked(compute_backend)?;
                        }
                        TemporalMixingLayer::Attention(_) | TemporalMixingLayer::Titans(_) => {
                            unreachable!(
                                "non-attention and non-titans GPU temporal mixer expected in strict setup"
                            );
                        }
                    }
                }

                self.compute_backend = compute_backend;
                return Ok(());
            }

            #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
            {
                return Err(crate::common::errors::ModelError::Backend {
                    message: format!(
                        "SharedTemporalProcessing requested GPU backend '{}' but this binary was built without GPU features.",
                        compute_backend.as_str()
                    ),
                });
            }
        }

        // CPU backend or non-GPU feature build
        self.gpu_device = None;
        self.temporal_mixing.set_compute_backend(compute_backend);
        self.compute_backend = compute_backend;
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

    /// Get runtime compute backend.
    #[inline]
    pub fn compute_backend(&self) -> ComputeBackend {
        self.compute_backend
    }

    /// Get a zero-copy reference to the underlying attention module if available.
    #[inline]
    pub fn attention(&self) -> Option<&PolyAttention> {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => Some(attn.as_ref()),
            _ => None,
        }
    }

    /// Get a mutable zero-copy reference to the underlying attention module if available.
    #[inline]
    pub fn attention_mut(&mut self) -> Option<&mut PolyAttention> {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => Some(attn.as_mut()),
            _ => None,
        }
    }

    /// Returns true when the temporal mixer is attention-based.
    #[inline]
    pub fn is_attention(&self) -> bool {
        matches!(self.temporal_mixing, TemporalMixingLayer::Attention(_))
    }

    /// Adapt attention window size using prediction norm thresholds.
    ///
    /// Returns the updated window size when attention metrics are available.
    /// For non-attention mixers, this is a no-op and returns `None`.
    #[inline]
    pub fn update_attention_window_from_pred_norm(
        &mut self,
        current_window: usize,
        min_window: usize,
        max_window: usize,
        step_up: usize,
        step_down: usize,
        pred_up: f32,
        pred_down: f32,
    ) -> Option<usize> {
        let min_window = min_window.max(1);
        let max_window = max_window.max(min_window);

        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => {
                let pred_norm = attn.last_pred_norm?;
                let mut window = current_window.clamp(min_window, max_window);
                if pred_norm > pred_up {
                    window = window.saturating_add(step_up).min(max_window);
                } else if pred_norm < pred_down {
                    window = window.saturating_sub(step_down).max(min_window);
                }
                self.window_size = Some(window);
                attn.set_window_size(Some(window));
                Some(window)
            }
            _ => None,
        }
    }

    #[inline]
    fn normalized_head_activity_ratio(avg_active_heads: Option<f32>, num_heads: usize) -> f32 {
        if let Some(avg) = avg_active_heads {
            (avg / (num_heads as f32).max(1.0)).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Get all head-activity outputs with a single temporal-mixer dispatch.
    #[inline]
    pub fn head_activity_summary(&self) -> HeadActivitySummary<'_> {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => HeadActivitySummary {
                ratio: Self::normalized_head_activity_ratio(
                    attn.last_avg_active_heads,
                    attn.num_heads,
                ),
                head_activity: attn.last_head_activity_vec.as_deref(),
                token_head_activity: attn.last_token_head_activity_vec.as_deref(),
            },
            TemporalMixingLayer::RgLruMoH(rglru) => HeadActivitySummary {
                ratio: Self::normalized_head_activity_ratio(
                    rglru.last_avg_active_heads,
                    rglru.num_heads,
                ),
                head_activity: rglru.last_head_activity_vec.as_deref(),
                token_head_activity: rglru.last_token_head_activity_vec.as_deref(),
            },
            TemporalMixingLayer::MambaMoH(m) => HeadActivitySummary {
                ratio: Self::normalized_head_activity_ratio(m.last_avg_active_heads, m.num_heads),
                head_activity: m.last_head_activity_vec.as_deref(),
                token_head_activity: m.last_token_head_activity_vec.as_deref(),
            },
            TemporalMixingLayer::Mamba2MoH(m) => HeadActivitySummary {
                ratio: Self::normalized_head_activity_ratio(m.last_avg_active_heads, m.num_heads),
                head_activity: m.last_head_activity_vec.as_deref(),
                token_head_activity: m.last_token_head_activity_vec.as_deref(),
            },
            TemporalMixingLayer::Titans(mac) => HeadActivitySummary {
                ratio: Self::normalized_head_activity_ratio(
                    mac.core.last_avg_active_heads,
                    mac.core.num_heads,
                ),
                head_activity: mac.core.last_head_activity_vec.as_deref(),
                token_head_activity: mac.core.last_token_head_activity_vec.as_deref(),
            },
            _ => HeadActivitySummary {
                ratio: 1.0,
                head_activity: None,
                token_head_activity: None,
            },
        }
    }

    /// Get head activity metrics if available (for MoH-based mixing)
    ///
    /// Uses shared accessor pattern with type-specific field access.
    pub fn get_head_activity_metrics(&self) -> (Option<f32>, Option<&[f32]>) {
        let summary = self.head_activity_summary();
        (Some(summary.ratio), summary.head_activity)
    }

    /// Get token head activity vector if available
    ///
    /// Uses shared accessor pattern with zero-copy view returns.
    pub fn get_token_head_activity_vec(&self) -> Option<&[f32]> {
        self.head_activity_summary().token_head_activity
    }

    /// Consume tau-range telemetry from temporal mixers that expose MoH stats.
    ///
    /// Returns `None` for temporal mixers without MoH telemetry.
    #[inline]
    pub fn take_tau_metrics(&mut self) -> Option<(f32, f32)> {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => attn.take_tau_metrics(),
            TemporalMixingLayer::RgLruMoH(rglru) => rglru.take_tau_metrics(),
            TemporalMixingLayer::MambaMoH(mamba) => mamba.take_tau_metrics(),
            TemporalMixingLayer::Mamba2MoH(mamba2) => mamba2.take_tau_metrics(),
            _ => None,
        }
    }

    /// Consume prediction-norm telemetry from temporal mixers that expose MoH stats.
    ///
    /// Returns `None` for temporal mixers without MoH telemetry.
    #[inline]
    pub fn take_pred_norm(&mut self) -> Option<f32> {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => attn.take_pred_norm(),
            TemporalMixingLayer::RgLruMoH(rglru) => rglru.take_pred_norm(),
            TemporalMixingLayer::MambaMoH(mamba) => mamba.take_pred_norm(),
            TemporalMixingLayer::Mamba2MoH(mamba2) => mamba2.take_pred_norm(),
            _ => None,
        }
    }

    /// Consume per-head activity counters from temporal mixers with MoH gating.
    ///
    /// Returns an empty vector for mixers that do not expose per-head telemetry.
    #[inline]
    pub fn get_head_metrics_and_reset(&mut self) -> Vec<(f32, usize)> {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => attn.get_head_metrics_and_reset(),
            TemporalMixingLayer::RgLruMoH(rglru) => rglru.get_head_metrics_and_reset(),
            TemporalMixingLayer::MambaMoH(mamba) => mamba.get_head_metrics_and_reset(),
            TemporalMixingLayer::Mamba2MoH(mamba2) => mamba2.get_head_metrics_and_reset(),
            _ => Vec::new(),
        }
    }

    /// Consume all available MoH telemetry in a single enum dispatch.
    #[inline]
    pub fn take_moh_metrics(&mut self) -> TemporalMoeMetrics {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => TemporalMoeMetrics {
                tau_metrics: attn.take_tau_metrics(),
                pred_norm: attn.take_pred_norm(),
                per_head_metrics: attn.get_head_metrics_and_reset(),
            },
            TemporalMixingLayer::RgLruMoH(rglru) => TemporalMoeMetrics {
                tau_metrics: rglru.take_tau_metrics(),
                pred_norm: rglru.take_pred_norm(),
                per_head_metrics: rglru.get_head_metrics_and_reset(),
            },
            TemporalMixingLayer::MambaMoH(mamba) => TemporalMoeMetrics {
                tau_metrics: mamba.take_tau_metrics(),
                pred_norm: mamba.take_pred_norm(),
                per_head_metrics: mamba.get_head_metrics_and_reset(),
            },
            TemporalMixingLayer::Mamba2MoH(mamba2) => TemporalMoeMetrics {
                tau_metrics: mamba2.take_tau_metrics(),
                pred_norm: mamba2.take_pred_norm(),
                per_head_metrics: mamba2.get_head_metrics_and_reset(),
            },
            _ => TemporalMoeMetrics::default(),
        }
    }

    /// Apply degree adaptation when the temporal mixer supports it.
    ///
    /// Currently only attention-based temporal mixers expose adaptive polynomial degree.
    #[inline]
    pub fn adapt_attention_degree(&mut self, metrics: &DegreeAdaptationMetrics) {
        if let TemporalMixingLayer::Attention(attn) = &mut self.temporal_mixing {
            attn.adapt_degree(metrics);
        }
    }

    /// Get window entropy metrics if available (for attention-based mixing)
    pub fn get_window_entropy(&self) -> Option<f32> {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => {
                if let Some((tmin, tmax)) = attn.last_tau_metrics {
                    let tau_span = (tmax - tmin).abs().max(0.0);
                    let pred_rms = attn.last_pred_norm.unwrap_or(0.0).max(0.0);
                    Some((0.7 * tau_span + 0.3 * pred_rms).clamp(0.0, 1.0))
                } else {
                    Some(0.0)
                }
            }
            _ => None,
        }
    }

    /// Apply MoH verification overrides for SSM MoH variants.
    ///
    /// This is a no-op for non-MoH temporal mixers.
    #[inline]
    pub fn set_moh_verification_overrides(&mut self, overrides: Option<Vec<f64>>) {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::RgLruMoH(rglru) => rglru.set_verification_overrides(overrides),
            TemporalMixingLayer::MambaMoH(mamba) => mamba.set_verification_overrides(overrides),
            TemporalMixingLayer::Mamba2MoH(mamba2) => mamba2.set_verification_overrides(overrides),
            _ => {}
        }
    }

    /// Forward step for streaming inference (token-by-token)
    pub fn forward_step_into(
        &mut self,
        input: &ndarray::ArrayView1<f32>,
        output: &mut ndarray::Array1<f32>,
    ) {
        // Streaming step mode uses CPU path
        self.temporal_mixing.forward_step_into(input, output);
    }

    /// GPU-accelerated forward pass for temporal processing.
    ///
    /// Executes the temporal mixing on GPU using UnifiedGpuKernels.
    /// This method handles upload, kernel dispatch, and download transparently.
    ///
    /// # Arguments
    /// * `input` - Input tensor on CPU (batch_size, embed_dim)
    ///
    /// # Returns
    /// * `Ok(output)` - Output tensor on CPU after GPU computation
    /// * `Err` if GPU operation fails
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let (seq_len, embed_dim) = input.dim();

        tracing::debug!(
            "GPU temporal forward: seq_len={}, embed_dim={}, mixer={}",
            seq_len,
            embed_dim,
            self.layer_type()
        );

        // Delegate to temporal variant-local GPU dispatch and backend caches.
        self.temporal_mixing.forward_gpu(input)
    }

    /// Compute gradients using the Layer trait
    pub fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            return self
                .compute_gradients_gpu(input, output_grads)
                .unwrap_or_else(|err| {
                    panic!("SharedTemporalProcessing GPU backward failed: {err}")
                });
        }

        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        if self.compute_backend.is_gpu() {
            panic!(
                "SharedTemporalProcessing configured for GPU backend '{}' but this binary has no GPU features enabled.",
                self.compute_backend.as_str()
            );
        }

        self.temporal_mixing.compute_gradients(input, output_grads)
    }

    /// Compute gradients through GPU-aware temporal mixing dispatch.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn compute_gradients_gpu(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        self.temporal_mixing
            .compute_gradients_gpu(input, output_grads)
    }

    /// Get a reference to the underlying temporal mixing layer
    ///
    /// This provides direct access for pattern matching and type-specific operations
    pub fn inner(&self) -> &TemporalMixingLayer {
        &self.temporal_mixing
    }

    /// Get a mutable reference to the underlying temporal mixing layer
    ///
    /// This provides direct mutable access for type-specific operations
    pub fn inner_mut(&mut self) -> &mut TemporalMixingLayer {
        &mut self.temporal_mixing
    }
}

impl crate::domain::layers::components::gradient_router::GradientRoutable
    for SharedTemporalProcessing
{
    fn gradient_count(&self) -> usize {
        self.temporal_mixing.parameters()
    }

    fn weight_norm(&self) -> f32 {
        self.temporal_mixing.weight_norm()
    }

    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::common::errors::Result<()> {
        self.temporal_mixing
            .apply_gradients(gradients, learning_rate)
    }
}

// ============================================================================
// GPU Component Trait Implementation (Phase 5.6)
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuComponent for SharedTemporalProcessing {
    /// Attach a pre-configured GPU device
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.gpu_device = Some(device.clone());
        // Also set device on the underlying temporal mixing layer
        self.temporal_mixing.set_gpu_device(device);
    }

    /// Enable GPU with automatic detection (strict no-fallback)
    fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        SharedTemporalProcessing::enable_gpu_auto_detect(self)
    }

    /// Check if GPU is ready for execution
    fn is_gpu_ready(&self) -> bool {
        if !self.compute_backend.is_gpu() {
            return false;
        }

        if self.uses_shared_gpu_device() {
            self.gpu_device.is_some()
        } else if self.uses_variant_local_gpu_backend() {
            true
        } else {
            false
        }
    }

    /// Get the GPU backend name if attached
    fn gpu_backend_name(&self) -> Option<&'static str> {
        if !self.compute_backend.is_gpu() {
            return None;
        }

        if self.uses_shared_gpu_device() {
            if let Some(device_arc) = &self.gpu_device {
                if let Ok(device) = device_arc.lock() {
                    return Some(device.backend().as_str());
                }
            }
            None
        } else if self.uses_variant_local_gpu_backend() {
            Some(self.compute_backend.as_str())
        } else {
            None
        }
    }

    /// Ensure buffers have sufficient capacity
    fn ensure_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        seq_len: usize,
    ) -> Result<()> {
        // Ensure GPU device has capacity for temporal operations
        if let Some(device_arc) = &self.gpu_device {
            let mut device =
                device_arc
                    .lock()
                    .map_err(|_| crate::common::errors::ModelError::Backend {
                        message: "Failed to acquire GPU device lock".to_string(),
                    })?;

            // Allocate buffers for temporal operations
            let input_size = batch_size * seq_len * embed_dim;
            let output_size = batch_size * seq_len * embed_dim;

            let _ = device.allocate_f32(input_size);
            let _ = device.allocate_f32(output_size);

            // For attention-based mixing, allocate attention score buffer
            let attention_scores_size =
                batch_size * self.temporal_mixing.num_heads() * seq_len * seq_len;
            let _ = device.allocate_f32(attention_scores_size);
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
    use super::*;
    use crate::domain::{compute_backend::ComputeBackend, models::config::TemporalMixingType};
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    use std::sync::{Arc, Mutex};

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    use crate::domain::{
        compute::GpuDevice, compute_backend::resolve_compute_backend_strict_auto_gpu,
    };

    #[test]
    fn test_shared_temporal_processing_layer_type() {
        // This test verifies that the SharedTemporalProcessing correctly
        // delegates to Layer trait methods
        let config = crate::domain::layers::components::common::CommonLayerConfig {
            embed_dim: 16,
            hidden_dim: 32,
            num_heads: 4,
            poly_degree: 3,
            max_pos: 32,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: crate::domain::models::config::TitanMemoryConfig::default(),
            temporal_mixing: TemporalMixingType::Attention,
        };

        let layers = crate::domain::layers::components::common::CommonLayers::new(&config);
        let stp = SharedTemporalProcessing::new(layers.temporal_mixing, None, false);

        assert_eq!(stp.layer_type(), "Attention");
        assert!(stp.parameters() > 0);
    }

    #[test]
    fn test_layer_trait_delegation() {
        // Test that Layer trait methods are correctly delegated
        let config = crate::domain::layers::components::common::CommonLayerConfig {
            embed_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            poly_degree: 3,
            max_pos: 16,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: crate::domain::models::config::TitanMemoryConfig::default(),
            temporal_mixing: TemporalMixingType::Attention,
        };

        let layers = crate::domain::layers::components::common::CommonLayers::new(&config);
        let mut stp = SharedTemporalProcessing::new(layers.temporal_mixing, None, false);

        // Test forward pass through Layer trait
        let input = Array2::zeros((2, 8));
        let output = stp.forward(&input);
        assert_eq!(output.dim(), (2, 8));

        // Test that parameters() returns consistent value
        let params = stp.parameters();
        assert!(params > 0);

        // Test weight_norm through Layer trait
        let norm = stp.weight_norm();
        assert!(norm >= 0.0);
    }

    #[test]
    fn test_shared_temporal_processing_set_compute_backend_checked_cpu() {
        let config = crate::domain::layers::components::common::CommonLayerConfig {
            embed_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            poly_degree: 3,
            max_pos: 16,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: crate::domain::models::config::TitanMemoryConfig::default(),
            temporal_mixing: TemporalMixingType::Attention,
        };

        let layers = crate::domain::layers::components::common::CommonLayers::new(&config);
        let mut stp = SharedTemporalProcessing::new(layers.temporal_mixing, None, false);

        stp.set_compute_backend_checked(ComputeBackend::Cpu)
            .expect("CPU backend should always be accepted");
        assert_eq!(stp.compute_backend(), ComputeBackend::Cpu);
    }

    #[test]
    fn test_shared_temporal_processing_set_compute_backend_checked_mamba2_moh_cpu() {
        let head_selection =
            crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 };
        let temporal = TemporalMixingLayer::Mamba2MoH(Box::new(
            crate::domain::layers::ssm::MoHMamba2::new(12, 3, &head_selection),
        ));
        let mut stp = SharedTemporalProcessing::new(temporal, None, false);

        stp.set_compute_backend_checked(ComputeBackend::Cpu)
            .expect("CPU backend should always be accepted for Mamba2MoH");
        assert_eq!(stp.compute_backend(), ComputeBackend::Cpu);
    }

    #[test]
    fn test_shared_temporal_processing_set_compute_backend_checked_mamba2_moh_gpu_strict() {
        let head_selection =
            crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 };
        let temporal = TemporalMixingLayer::Mamba2MoH(Box::new(
            crate::domain::layers::ssm::MoHMamba2::new(12, 3, &head_selection),
        ));
        let mut stp = SharedTemporalProcessing::new(temporal, None, false);

        let result = stp.set_compute_backend_checked(ComputeBackend::Vulkan);
        match result {
            Ok(()) => assert!(stp.compute_backend().is_gpu()),
            Err(err) => {
                let msg = format!("{}", err).to_lowercase();
                assert!(
                    msg.contains("without gpu features")
                        || msg.contains("unavailable")
                        || msg.contains("gpu")
                        || msg.contains("backend"),
                    "expected strict GPU validation error, got: {}",
                    err
                );
            }
        }
    }

    #[test]
    fn test_shared_temporal_processing_set_compute_backend_checked_mamba_moh_cpu() {
        let head_selection =
            crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 };
        let temporal = TemporalMixingLayer::MambaMoH(Box::new(
            crate::domain::layers::ssm::MoHMamba::new(12, 3, &head_selection),
        ));
        let mut stp = SharedTemporalProcessing::new(temporal, None, false);

        stp.set_compute_backend_checked(ComputeBackend::Cpu)
            .expect("CPU backend should always be accepted for MambaMoH");
        assert_eq!(stp.compute_backend(), ComputeBackend::Cpu);
    }

    #[test]
    fn test_shared_temporal_processing_set_compute_backend_checked_mamba_moh_gpu_strict() {
        let head_selection =
            crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 };
        let temporal = TemporalMixingLayer::MambaMoH(Box::new(
            crate::domain::layers::ssm::MoHMamba::new(12, 3, &head_selection),
        ));
        let mut stp = SharedTemporalProcessing::new(temporal, None, false);

        let result = stp.set_compute_backend_checked(ComputeBackend::Vulkan);
        match result {
            Ok(()) => assert!(stp.compute_backend().is_gpu()),
            Err(err) => {
                let msg = format!("{}", err).to_lowercase();
                assert!(
                    msg.contains("without gpu features")
                        || msg.contains("unavailable")
                        || msg.contains("gpu")
                        || msg.contains("backend"),
                    "expected strict GPU validation error, got: {}",
                    err
                );
            }
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_shared_temporal_processing_gpu_readiness_attention_requires_device() {
        use crate::domain::compute::GpuComponent;

        let config = crate::domain::layers::components::common::CommonLayerConfig {
            embed_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            poly_degree: 3,
            max_pos: 16,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: crate::domain::models::config::TitanMemoryConfig::default(),
            temporal_mixing: TemporalMixingType::Attention,
        };

        let layers = crate::domain::layers::components::common::CommonLayers::new(&config);
        let mut stp = SharedTemporalProcessing::new(layers.temporal_mixing, None, false);
        stp.compute_backend = ComputeBackend::Vulkan;
        stp.gpu_device = None;

        assert!(!stp.is_gpu_ready());
        assert_eq!(stp.gpu_backend_name(), None);
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_shared_temporal_processing_gpu_readiness_ssm_without_device() {
        use crate::domain::compute::GpuComponent;

        let temporal =
            TemporalMixingLayer::Mamba(Box::new(crate::domain::layers::ssm::Mamba::new(8)));
        let mut stp = SharedTemporalProcessing::new(temporal, None, false);
        stp.compute_backend = ComputeBackend::Vulkan;
        stp.gpu_device = None;

        assert!(stp.is_gpu_ready());
        assert_eq!(stp.gpu_backend_name(), Some("vulkan"));
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_shared_temporal_processing_reuses_preattached_gpu_device_for_attention() {
        let backend = match resolve_compute_backend_strict_auto_gpu() {
            Ok(backend) => backend,
            Err(_) => return,
        };

        let config = crate::domain::layers::components::common::CommonLayerConfig {
            embed_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            poly_degree: 3,
            max_pos: 16,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: crate::domain::models::config::TitanMemoryConfig::default(),
            temporal_mixing: TemporalMixingType::Attention,
        };
        let layers = crate::domain::layers::components::common::CommonLayers::new(&config);
        let mut stp = SharedTemporalProcessing::new(layers.temporal_mixing, None, false);

        let device = Arc::new(Mutex::new(
            GpuDevice::new(backend).expect("resolved backend should create a GPU device"),
        ));
        stp.gpu_device = Some(device.clone());

        stp.set_compute_backend_checked(backend)
            .expect("pre-attached matching GPU device should be reused");

        let attached = stp
            .gpu_device
            .as_ref()
            .expect("GPU device should remain attached")
            .clone();
        assert!(Arc::ptr_eq(&attached, &device));
    }
}

// ============================================================================
// GPU Component Implementation (Phase 5.6)
// ============================================================================
