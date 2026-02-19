//! Unified layer workspace for consolidated memory management.
//!
//! This module provides a single workspace type that consolidates the various
//! buffer pools used across TransformerBlock, DiffusionBlock, and SSM layers.
//! It replaces the separate IntermediateBufferPool, AdaptiveResidualsWorkspace,
//! and workspace-specific implementations with a single, coherent design.
//!
//! ## GPU Support (Phase 5.2)
//!
//! The workspace supports both CPU and GPU memory management:
//! - CPU buffers: ndarray Array2 for standard operations
//! - GPU buffers: GpuBuffer handles for GPU-accelerated operations
//! - Automatic backend selection via `compute_backend` field

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::gpu_memory::GpuBuffer;
use crate::domain::compute_backend::ComputeBackend;

use super::workspace_managed::{WorkspaceManaged, WorkspaceStats};

/// Unified workspace for a single layer (Transformer, Diffusion, or SSM).
///
/// Consolidates:
/// - Normalization buffers (norm1_out, norm2_out)
/// - Temporal mixing outputs (attention_out, ssm_out)
/// - FFN buffers (ffn_intermediate)
/// - Residual computation buffers
/// - Optional streaming state (for SSM/RG-LRU)
/// - Diffusion-specific buffers (time embeddings, FiLM modulation)
///
/// The workspace uses power-of-2 capacity sizing to minimize reallocations
/// across varying sequence lengths and batch sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedLayerWorkspace {
    // --- Core buffers (used by all block types) ---
    /// Output of first normalization
    #[serde(skip)]
    norm1_out: Option<Array2<f32>>,

    /// Output of temporal mixing (attention, SSM, etc.)
    #[serde(skip)]
    temporal_out: Option<Array2<f32>>,

    /// Residual computation buffer
    #[serde(skip)]
    residual1: Option<Array2<f32>>,

    /// Output of second normalization
    #[serde(skip)]
    norm2_out: Option<Array2<f32>>,

    /// FFN intermediate buffer
    #[serde(skip)]
    ffn_intermediate: Option<Array2<f32>>,

    /// FFN output buffer (reused)
    #[serde(skip)]
    ffn_out: Option<Array2<f32>>,

    // --- Streaming state (SSM/RG-LRU only) ---
    /// RNN state for recurrent layers
    #[serde(skip)]
    streaming_state: Option<Array2<f32>>,

    /// Context matrix for attention-based mixing
    #[serde(skip)]
    context_buffer: Option<Array2<f32>>,

    // --- Diffusion-specific buffers ---
    /// Input cache (consolidates input_original and input_used from DiffusionCachedIntermediates)
    #[serde(skip)]
    input_buffer: Option<Array2<f32>>,

    /// Time embedding from diffusion timestep embedding
    #[serde(skip)]
    time_embed: Option<Array1<f32>>,

    /// FiLM scale parameter (gamma): [batch, 4*embed_dim] for (gamma_attn, beta_attn, gamma_ffn, beta_ffn)
    #[serde(skip)]
    film_modulation_scale: Option<Array2<f32>>,

    /// FiLM shift parameter (beta): alias for film_modulation_scale to support scalar operations
    #[serde(skip)]
    film_modulation_shift: Option<Array2<f32>>,

    /// Final output buffer for Diffusion blocks (stores network output before EDM scaling)
    #[serde(skip)]
    output_buffer: Option<Array2<f32>>,

    // --- GPU Buffers (Phase 5.2) ---
    /// GPU buffer for norm1_out (when using GPU backend)
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[serde(skip)]
    gpu_norm1_out: Option<GpuBuffer>,

    /// GPU buffer for temporal_out (when using GPU backend)
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[serde(skip)]
    gpu_temporal_out: Option<GpuBuffer>,

    /// GPU buffer for residual1 (when using GPU backend)
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[serde(skip)]
    gpu_residual1: Option<GpuBuffer>,

    /// GPU buffer for norm2_out (when using GPU backend)
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[serde(skip)]
    gpu_norm2_out: Option<GpuBuffer>,

    /// GPU buffer for ffn_out (when using GPU backend)
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[serde(skip)]
    gpu_ffn_out: Option<GpuBuffer>,

    /// GPU buffer for attention context matrix (embed_dim x embed_dim).
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[serde(skip)]
    gpu_context_buffer: Option<GpuBuffer>,

    /// Current power-of-2 capacity (in f32 elements) for GPU core buffers.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[serde(skip)]
    gpu_core_capacity_elements: usize,

    // --- Metadata ---
    /// Expected shape after next allocation
    expected_shape: Option<(usize, usize)>,

    /// Last embedding dimension used for allocation.
    expected_embed_dim: Option<usize>,

    /// Current allocation limit to prevent unbounded growth
    allocation_limit: usize,

    /// Number of allocations made
    allocation_count: u32,

    /// Runtime compute backend associated with this workspace.
    #[serde(default)]
    compute_backend: ComputeBackend,
}

impl Default for UnifiedLayerWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedLayerWorkspace {
    /// Create a new empty workspace.
    pub fn new() -> Self {
        Self::new_with_backend(ComputeBackend::Cpu)
    }

    /// Create a new workspace pinned to a specific compute backend.
    pub fn new_with_backend(compute_backend: ComputeBackend) -> Self {
        Self {
            norm1_out: None,
            temporal_out: None,
            residual1: None,
            norm2_out: None,
            ffn_intermediate: None,
            ffn_out: None,
            streaming_state: None,
            context_buffer: None,
            input_buffer: None,
            time_embed: None,
            film_modulation_scale: None,
            film_modulation_shift: None,
            output_buffer: None,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            gpu_norm1_out: None,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            gpu_temporal_out: None,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            gpu_residual1: None,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            gpu_norm2_out: None,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            gpu_ffn_out: None,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            gpu_context_buffer: None,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            gpu_core_capacity_elements: 0,
            expected_shape: None,
            expected_embed_dim: None,
            allocation_limit: 512 * 1024 * 1024, // 512 MB by default
            allocation_count: 0,
            compute_backend,
        }
    }

    /// Set compute backend metadata for this workspace.
    #[inline]
    pub fn set_compute_backend(&mut self, compute_backend: ComputeBackend) {
        self.compute_backend = compute_backend;
    }

    /// Get compute backend metadata for this workspace.
    #[inline]
    pub fn compute_backend(&self) -> ComputeBackend {
        self.compute_backend
    }

    /// Set the allocation limit to prevent runaway memory usage.
    pub fn set_allocation_limit(&mut self, limit: usize) {
        self.allocation_limit = limit;
    }

    /// Ensure exact-shape execution buffers for hot forward paths.
    ///
    /// Unlike `ensure_capacity`, this method allocates exact shapes (no power-of-2 rounding)
    /// so in-place kernels can write directly into workspace buffers without slice adapters.
    pub fn ensure_exact_execution_capacity(
        &mut self,
        batch_size: usize,
        seq_len: usize,
        embed_dim: usize,
    ) {
        let batch_size = batch_size.max(1);
        let seq_len = seq_len.max(1);
        let mut changed = false;

        let estimated_bytes = estimate_allocation(
            batch_size,
            seq_len,
            embed_dim,
            self.streaming_state.is_some(),
            self.context_buffer.is_some(),
            self.input_buffer.is_some(),
            self.time_embed.is_some(),
            self.film_modulation_scale.is_some(),
            self.film_modulation_shift.is_some(),
            self.output_buffer.is_some(),
        );
        if estimated_bytes > self.allocation_limit {
            eprintln!(
                "WARNING: Workspace exact execution allocation ({} bytes) exceeds limit ({} bytes). Proceeding with caution.",
                estimated_bytes, self.allocation_limit
            );
        }

        changed |= ensure_array2_shape(&mut self.norm1_out, batch_size, seq_len);
        changed |= ensure_array2_shape(&mut self.temporal_out, batch_size, seq_len);
        changed |= ensure_array2_shape(&mut self.residual1, batch_size, seq_len);
        changed |= ensure_array2_shape(&mut self.norm2_out, batch_size, seq_len);
        changed |= ensure_array2_shape(&mut self.ffn_intermediate, batch_size, seq_len);
        changed |= ensure_array2_shape(&mut self.ffn_out, batch_size, seq_len);

        if self.streaming_state.is_some() {
            changed |= ensure_array2_shape(&mut self.streaming_state, batch_size, embed_dim);
        }
        if self.context_buffer.is_some() {
            changed |= ensure_array2_shape(&mut self.context_buffer, embed_dim, embed_dim);
        }
        if self.input_buffer.is_some() {
            changed |= ensure_array2_shape(&mut self.input_buffer, batch_size, seq_len);
        }
        if self.time_embed.is_some() {
            changed |= ensure_array1_len(&mut self.time_embed, embed_dim);
        }
        if self.film_modulation_scale.is_some() {
            changed |=
                ensure_array2_shape(&mut self.film_modulation_scale, batch_size, 4 * embed_dim);
        }
        if self.film_modulation_shift.is_some() {
            changed |=
                ensure_array2_shape(&mut self.film_modulation_shift, batch_size, 4 * embed_dim);
        }
        if self.output_buffer.is_some() {
            changed |= ensure_array2_shape(&mut self.output_buffer, batch_size, seq_len);
        }

        self.expected_shape = Some((batch_size, seq_len));
        self.expected_embed_dim = Some(embed_dim);
        if changed {
            self.allocation_count = self.allocation_count.saturating_add(1);
        }
    }

    /// Get the number of allocations made.
    pub fn allocation_count(&self) -> u32 {
        self.allocation_count
    }

    /// Enable or disable the optional streaming-state buffer.
    pub fn set_streaming_state_enabled(&mut self, enabled: bool) {
        self.streaming_state = enabled.then(|| Array2::zeros((0, 0)));
    }

    /// Enable or disable the optional context buffer.
    pub fn set_context_buffer_enabled(&mut self, enabled: bool) {
        self.context_buffer = enabled.then(|| Array2::zeros((0, 0)));
    }

    /// Enable or disable diffusion-specific optional buffers.
    pub fn set_diffusion_buffers_enabled(&mut self, enabled: bool) {
        if enabled {
            self.input_buffer = Some(Array2::zeros((0, 0)));
            self.time_embed = Some(Array1::zeros(0));
            self.film_modulation_scale = Some(Array2::zeros((0, 0)));
            self.film_modulation_shift = Some(Array2::zeros((0, 0)));
            self.output_buffer = Some(Array2::zeros((0, 0)));
        } else {
            self.input_buffer = None;
            self.time_embed = None;
            self.film_modulation_scale = None;
            self.film_modulation_shift = None;
            self.output_buffer = None;
        }
    }

    // --- Accessors ---

    #[inline]
    pub fn norm1_out(&self) -> Option<&Array2<f32>> {
        self.norm1_out.as_ref()
    }

    #[inline]
    pub fn norm1_out_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.norm1_out.as_mut()
    }

    #[inline]
    pub fn temporal_out(&self) -> Option<&Array2<f32>> {
        self.temporal_out.as_ref()
    }

    #[inline]
    pub fn temporal_out_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.temporal_out.as_mut()
    }

    #[inline]
    pub fn residual1(&self) -> Option<&Array2<f32>> {
        self.residual1.as_ref()
    }

    #[inline]
    pub fn residual1_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.residual1.as_mut()
    }

    #[inline]
    pub fn norm2_out(&self) -> Option<&Array2<f32>> {
        self.norm2_out.as_ref()
    }

    #[inline]
    pub fn norm2_out_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.norm2_out.as_mut()
    }

    #[inline]
    pub fn ffn_intermediate(&self) -> Option<&Array2<f32>> {
        self.ffn_intermediate.as_ref()
    }

    #[inline]
    pub fn ffn_intermediate_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.ffn_intermediate.as_mut()
    }

    #[inline]
    pub fn ffn_out(&self) -> Option<&Array2<f32>> {
        self.ffn_out.as_ref()
    }

    #[inline]
    pub fn ffn_out_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.ffn_out.as_mut()
    }

    #[inline]
    pub fn streaming_state(&self) -> Option<&Array2<f32>> {
        self.streaming_state.as_ref()
    }

    #[inline]
    pub fn streaming_state_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.streaming_state.as_mut()
    }

    #[inline]
    pub fn context_buffer(&self) -> Option<&Array2<f32>> {
        self.context_buffer.as_ref()
    }

    #[inline]
    pub fn context_buffer_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.context_buffer.as_mut()
    }

    #[inline]
    pub fn input_buffer(&self) -> Option<&Array2<f32>> {
        self.input_buffer.as_ref()
    }

    #[inline]
    pub fn input_buffer_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.input_buffer.as_mut()
    }

    #[inline]
    pub fn time_embed(&self) -> Option<&Array1<f32>> {
        self.time_embed.as_ref()
    }

    #[inline]
    pub fn time_embed_mut(&mut self) -> Option<&mut Array1<f32>> {
        self.time_embed.as_mut()
    }

    #[inline]
    pub fn film_modulation_scale(&self) -> Option<&Array2<f32>> {
        self.film_modulation_scale.as_ref()
    }

    #[inline]
    pub fn film_modulation_scale_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.film_modulation_scale.as_mut()
    }

    #[inline]
    pub fn film_modulation_shift(&self) -> Option<&Array2<f32>> {
        self.film_modulation_shift.as_ref()
    }

    #[inline]
    pub fn film_modulation_shift_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.film_modulation_shift.as_mut()
    }

    #[inline]
    pub fn output_buffer(&self) -> Option<&Array2<f32>> {
        self.output_buffer.as_ref()
    }

    #[inline]
    pub fn output_buffer_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.output_buffer.as_mut()
    }

    /// Set GPU core buffer handles for backend execution.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn set_gpu_core_buffers(
        &mut self,
        norm1_out: Option<GpuBuffer>,
        temporal_out: Option<GpuBuffer>,
        residual1: Option<GpuBuffer>,
        norm2_out: Option<GpuBuffer>,
        ffn_out: Option<GpuBuffer>,
    ) {
        self.gpu_norm1_out = norm1_out;
        self.gpu_temporal_out = temporal_out;
        self.gpu_residual1 = residual1;
        self.gpu_norm2_out = norm2_out;
        self.gpu_ffn_out = ffn_out;
        self.refresh_gpu_core_capacity();
    }

    /// Get GPU core buffer handles for backend execution.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn gpu_core_buffers(
        &self,
    ) -> (
        Option<GpuBuffer>,
        Option<GpuBuffer>,
        Option<GpuBuffer>,
        Option<GpuBuffer>,
        Option<GpuBuffer>,
    ) {
        (
            self.gpu_norm1_out,
            self.gpu_temporal_out,
            self.gpu_residual1,
            self.gpu_norm2_out,
            self.gpu_ffn_out,
        )
    }

    /// Check if all core buffers are allocated.
    pub fn all_buffers_allocated(&self) -> bool {
        self.norm1_out.is_some()
            && self.temporal_out.is_some()
            && self.residual1.is_some()
            && self.norm2_out.is_some()
            && self.ffn_intermediate.is_some()
            && self.ffn_out.is_some()
    }

    /// Estimate total memory usage of allocated buffers.
    pub fn estimate_memory_usage(&self) -> usize {
        let mut total = 0;
        for buf in [
            &self.norm1_out,
            &self.temporal_out,
            &self.residual1,
            &self.norm2_out,
            &self.ffn_intermediate,
            &self.ffn_out,
            &self.streaming_state,
            &self.context_buffer,
            &self.input_buffer,
            &self.film_modulation_scale,
            &self.film_modulation_shift,
            &self.output_buffer,
        ] {
            if let Some(arr) = buf {
                total += arr.len() * std::mem::size_of::<f32>();
            }
        }
        // Time embedding (1D array)
        if let Some(te) = &self.time_embed {
            total += te.len() * std::mem::size_of::<f32>();
        }
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        for gpu in [
            self.gpu_norm1_out,
            self.gpu_temporal_out,
            self.gpu_residual1,
            self.gpu_norm2_out,
            self.gpu_ffn_out,
            self.gpu_context_buffer,
        ] {
            if let Some(buf) = gpu {
                total = total.saturating_add(buf.size_bytes());
            }
        }
        total
    }

    /// Reset streaming state between sequences.
    pub fn reset_streaming(&mut self) {
        if let Some(state) = self.streaming_state.as_mut() {
            state.fill(0.0);
        }
    }

    /// Clear all buffers to free memory.
    pub fn clear_all(&mut self) {
        self.norm1_out = None;
        self.temporal_out = None;
        self.residual1 = None;
        self.norm2_out = None;
        self.ffn_intermediate = None;
        self.ffn_out = None;
        self.streaming_state = None;
        self.context_buffer = None;
        self.input_buffer = None;
        self.time_embed = None;
        self.film_modulation_scale = None;
        self.film_modulation_shift = None;
        self.output_buffer = None;
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            self.gpu_norm1_out = None;
            self.gpu_temporal_out = None;
            self.gpu_residual1 = None;
            self.gpu_norm2_out = None;
            self.gpu_ffn_out = None;
            self.gpu_context_buffer = None;
            self.gpu_core_capacity_elements = 0;
        }
        self.expected_shape = None;
        self.expected_embed_dim = None;
    }

    // --- GPU Buffer Management (Phase 5.3) ---

    /// Ensure GPU buffers are allocated for the given capacity.
    ///
    /// This method allocates GPU buffers for core workspace fields when using
    /// a GPU backend. Uses power-of-2 sizing for efficient memory alignment.
    ///
    /// # Arguments
    /// * `gpu_device` - GPU device to allocate from
    /// * `batch_size` - Batch size (sequence count)
    /// * `seq_len` - Sequence length (tokens per sequence)
    /// * `embed_dim` - Embedding dimension
    ///
    /// # Returns
    /// Ok(()) on success, or an error if allocation fails
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn ensure_gpu_capacity(
        &mut self,
        gpu_device: &mut crate::domain::compute::GpuDevice,
        batch_size: usize,
        seq_len: usize,
        embed_dim: usize,
    ) -> crate::common::errors::Result<()> {
        let batch_size = batch_size.max(1);
        let seq_len = seq_len.max(1);

        // Use power-of-2 sizing for GPU buffers
        let alloc_batch = next_power_of_two(batch_size);
        let alloc_seq = next_power_of_two(seq_len);
        let required_elements = alloc_batch.saturating_mul(alloc_seq);
        let core_ready =
            self.gpu_buffers_allocated() && self.gpu_core_capacity_elements >= required_elements;

        if !core_ready {
            // Free previous core buffers before reallocating larger capacity.
            for buffer in [
                self.gpu_norm1_out.take(),
                self.gpu_temporal_out.take(),
                self.gpu_residual1.take(),
                self.gpu_norm2_out.take(),
                self.gpu_ffn_out.take(),
            ]
            .into_iter()
            .flatten()
            {
                gpu_device.deallocate(buffer);
            }

            self.gpu_norm1_out = Some(gpu_device.allocate_f32(required_elements)?);
            self.gpu_temporal_out = Some(gpu_device.allocate_f32(required_elements)?);
            self.gpu_residual1 = Some(gpu_device.allocate_f32(required_elements)?);
            self.gpu_norm2_out = Some(gpu_device.allocate_f32(required_elements)?);
            self.gpu_ffn_out = Some(gpu_device.allocate_f32(required_elements)?);
            self.gpu_core_capacity_elements = required_elements;
        }

        if self.context_buffer.is_some() {
            let alloc_embed = next_power_of_two(embed_dim.max(1));
            let required_context_elements = alloc_embed.saturating_mul(alloc_embed);
            let current_context_capacity = self
                .gpu_context_buffer
                .map(|buffer| buffer.size_f32())
                .unwrap_or(0);
            if current_context_capacity < required_context_elements {
                if let Some(existing) = self.gpu_context_buffer.take() {
                    gpu_device.deallocate(existing);
                }
                self.gpu_context_buffer = Some(gpu_device.allocate_f32(required_context_elements)?);
            }
        } else if let Some(existing) = self.gpu_context_buffer.take() {
            gpu_device.deallocate(existing);
        }

        Ok(())
    }

    /// Get GPU buffer for norm1_out
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn gpu_norm1_out(&self) -> Option<GpuBuffer> {
        self.gpu_norm1_out
    }

    /// Get GPU buffer for temporal_out
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn gpu_temporal_out(&self) -> Option<GpuBuffer> {
        self.gpu_temporal_out
    }

    /// Get GPU buffer for residual1
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn gpu_residual1(&self) -> Option<GpuBuffer> {
        self.gpu_residual1
    }

    /// Get GPU buffer for norm2_out
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn gpu_norm2_out(&self) -> Option<GpuBuffer> {
        self.gpu_norm2_out
    }

    /// Get GPU buffer for ffn_out
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn gpu_ffn_out(&self) -> Option<GpuBuffer> {
        self.gpu_ffn_out
    }

    /// Get GPU buffer for context matrix
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn gpu_context_buffer(&self) -> Option<GpuBuffer> {
        self.gpu_context_buffer
    }

    /// Set GPU buffer for norm1_out
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn set_gpu_norm1_out(&mut self, buffer: Option<GpuBuffer>) {
        self.gpu_norm1_out = buffer;
        self.refresh_gpu_core_capacity();
    }

    /// Set GPU buffer for temporal_out
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn set_gpu_temporal_out(&mut self, buffer: Option<GpuBuffer>) {
        self.gpu_temporal_out = buffer;
        self.refresh_gpu_core_capacity();
    }

    /// Set GPU buffer for residual1
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn set_gpu_residual1(&mut self, buffer: Option<GpuBuffer>) {
        self.gpu_residual1 = buffer;
        self.refresh_gpu_core_capacity();
    }

    /// Set GPU buffer for norm2_out
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn set_gpu_norm2_out(&mut self, buffer: Option<GpuBuffer>) {
        self.gpu_norm2_out = buffer;
        self.refresh_gpu_core_capacity();
    }

    /// Set GPU buffer for ffn_out
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn set_gpu_ffn_out(&mut self, buffer: Option<GpuBuffer>) {
        self.gpu_ffn_out = buffer;
        self.refresh_gpu_core_capacity();
    }

    /// Set GPU buffer for context matrix
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    pub fn set_gpu_context_buffer(&mut self, buffer: Option<GpuBuffer>) {
        self.gpu_context_buffer = buffer;
    }

    /// Clear GPU buffers only
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn clear_gpu_buffers(&mut self) {
        self.gpu_norm1_out = None;
        self.gpu_temporal_out = None;
        self.gpu_residual1 = None;
        self.gpu_norm2_out = None;
        self.gpu_ffn_out = None;
        self.gpu_context_buffer = None;
        self.gpu_core_capacity_elements = 0;
    }

    /// Clear GPU buffers and release the associated device memory.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn clear_gpu_buffers_with_device(
        &mut self,
        gpu_device: &mut crate::domain::compute::GpuDevice,
    ) {
        for buffer in [
            self.gpu_norm1_out.take(),
            self.gpu_temporal_out.take(),
            self.gpu_residual1.take(),
            self.gpu_norm2_out.take(),
            self.gpu_ffn_out.take(),
            self.gpu_context_buffer.take(),
        ]
        .into_iter()
        .flatten()
        {
            gpu_device.deallocate(buffer);
        }
        self.gpu_core_capacity_elements = 0;
    }

    /// Check if GPU buffers are allocated
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn gpu_buffers_allocated(&self) -> bool {
        self.gpu_norm1_out.is_some()
            && self.gpu_temporal_out.is_some()
            && self.gpu_residual1.is_some()
            && self.gpu_norm2_out.is_some()
            && self.gpu_ffn_out.is_some()
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    fn refresh_gpu_core_capacity(&mut self) {
        self.gpu_core_capacity_elements = [
            self.gpu_norm1_out,
            self.gpu_temporal_out,
            self.gpu_residual1,
            self.gpu_norm2_out,
            self.gpu_ffn_out,
        ]
        .into_iter()
        .flatten()
        .map(|buffer| buffer.size_f32())
        .min()
        .unwrap_or(0);
    }
}

impl WorkspaceManaged for UnifiedLayerWorkspace {
    fn ensure_capacity(&mut self, batch_size: usize, seq_len: usize, embed_dim: usize) {
        let batch_size = batch_size.max(1);
        let seq_len = seq_len.max(1);
        let mut changed = false;

        let needs_core_realloc = match self.expected_shape {
            None => true,
            Some((prev_batch, prev_seq)) => {
                let grow = batch_size > prev_batch || seq_len > prev_seq;
                // Shrink only when dimensions drop materially to avoid churn.
                let shrink = batch_size.saturating_mul(4) <= prev_batch
                    && seq_len.saturating_mul(4) <= prev_seq;
                let embed_changed = self
                    .expected_embed_dim
                    .is_some_and(|prev_embed| prev_embed != embed_dim);
                grow || shrink || embed_changed
            }
        };

        if needs_core_realloc {
            let alloc_batch = next_power_of_two(batch_size);
            let alloc_seq = next_power_of_two(seq_len);

            let estimated_bytes = estimate_allocation(
                alloc_batch,
                alloc_seq,
                embed_dim,
                self.streaming_state.is_some(),
                self.context_buffer.is_some(),
                self.input_buffer.is_some(),
                self.time_embed.is_some(),
                self.film_modulation_scale.is_some(),
                self.film_modulation_shift.is_some(),
                self.output_buffer.is_some(),
            );
            if estimated_bytes > self.allocation_limit {
                eprintln!(
                    "WARNING: Workspace allocation ({} bytes) exceeds limit ({} bytes). Proceeding with caution.",
                    estimated_bytes, self.allocation_limit
                );
            }

            changed |= ensure_array2_shape(&mut self.norm1_out, alloc_batch, alloc_seq);
            changed |= ensure_array2_shape(&mut self.temporal_out, alloc_batch, alloc_seq);
            changed |= ensure_array2_shape(&mut self.residual1, alloc_batch, alloc_seq);
            changed |= ensure_array2_shape(&mut self.norm2_out, alloc_batch, alloc_seq);
            changed |= ensure_array2_shape(&mut self.ffn_intermediate, alloc_batch, alloc_seq);
            changed |= ensure_array2_shape(&mut self.ffn_out, alloc_batch, alloc_seq);

            self.expected_shape = Some((alloc_batch, alloc_seq));
        }

        let (alloc_batch, alloc_seq) = self.expected_shape.unwrap_or((batch_size, seq_len));
        if self.streaming_state.is_some() {
            changed |= ensure_array2_shape(&mut self.streaming_state, alloc_batch, embed_dim);
        }
        if self.context_buffer.is_some() {
            changed |= ensure_array2_shape(&mut self.context_buffer, embed_dim, embed_dim);
        }
        if self.input_buffer.is_some() {
            changed |= ensure_array2_shape(&mut self.input_buffer, alloc_batch, alloc_seq);
        }
        if self.time_embed.is_some() {
            changed |= ensure_array1_len(&mut self.time_embed, embed_dim);
        }
        if self.film_modulation_scale.is_some() {
            changed |=
                ensure_array2_shape(&mut self.film_modulation_scale, alloc_batch, 4 * embed_dim);
        }
        if self.film_modulation_shift.is_some() {
            changed |=
                ensure_array2_shape(&mut self.film_modulation_shift, alloc_batch, 4 * embed_dim);
        }
        if self.output_buffer.is_some() {
            changed |= ensure_array2_shape(&mut self.output_buffer, alloc_batch, alloc_seq);
        }

        self.expected_embed_dim = Some(embed_dim);
        if changed {
            self.allocation_count = self.allocation_count.saturating_add(1);
        }
    }

    fn clear_workspace(&mut self) {
        self.clear_all();
    }

    fn workspace_stats(&self) -> WorkspaceStats {
        let buffer_count = [
            &self.norm1_out,
            &self.temporal_out,
            &self.residual1,
            &self.norm2_out,
            &self.ffn_intermediate,
            &self.ffn_out,
            &self.streaming_state,
            &self.context_buffer,
            &self.input_buffer,
            &self.film_modulation_scale,
            &self.film_modulation_shift,
            &self.output_buffer,
        ]
        .iter()
        .filter(|b| b.is_some())
        .count()
            + usize::from(self.time_embed.is_some());
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        let buffer_count = buffer_count
            + usize::from(self.gpu_norm1_out.is_some())
            + usize::from(self.gpu_temporal_out.is_some())
            + usize::from(self.gpu_residual1.is_some())
            + usize::from(self.gpu_norm2_out.is_some())
            + usize::from(self.gpu_ffn_out.is_some())
            + usize::from(self.gpu_context_buffer.is_some());

        WorkspaceStats {
            total_bytes: self.estimate_memory_usage(),
            buffer_count,
            expected_shape: self.expected_shape,
        }
    }
}

#[inline]
fn ensure_array2_shape(slot: &mut Option<Array2<f32>>, rows: usize, cols: usize) -> bool {
    match slot {
        Some(arr) if arr.dim() == (rows, cols) => false,
        _ => {
            *slot = Some(Array2::zeros((rows, cols)));
            true
        }
    }
}

#[inline]
fn ensure_array1_len(slot: &mut Option<Array1<f32>>, len: usize) -> bool {
    match slot {
        Some(arr) if arr.len() == len => false,
        _ => {
            *slot = Some(Array1::zeros(len));
            true
        }
    }
}

/// Compute next power of two for capacity allocation.
#[inline]
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n.checked_next_power_of_two().unwrap_or(n)
    }
}

/// Estimate total allocation in bytes for given dimensions.
#[inline]
fn estimate_allocation(
    batch: usize,
    seq: usize,
    embed_dim: usize,
    has_streaming_state: bool,
    has_context_buffer: bool,
    has_input_buffer: bool,
    has_time_embed: bool,
    has_film_scale: bool,
    has_film_shift: bool,
    has_output_buffer: bool,
) -> usize {
    let scalar_buffers = 6usize;
    let mut elements = scalar_buffers.saturating_mul(batch).saturating_mul(seq);

    if has_streaming_state {
        elements = elements.saturating_add(batch.saturating_mul(embed_dim));
    }
    if has_context_buffer {
        elements = elements.saturating_add(embed_dim.saturating_mul(embed_dim));
    }
    if has_input_buffer {
        elements = elements.saturating_add(batch.saturating_mul(seq));
    }
    if has_film_scale {
        elements = elements.saturating_add(batch.saturating_mul(embed_dim).saturating_mul(4));
    }
    if has_film_shift {
        elements = elements.saturating_add(batch.saturating_mul(embed_dim).saturating_mul(4));
    }
    if has_output_buffer {
        elements = elements.saturating_add(batch.saturating_mul(seq));
    }
    if has_time_embed {
        elements = elements.saturating_add(embed_dim);
    }

    elements.saturating_mul(std::mem::size_of::<f32>())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    use crate::domain::compute::gpu_memory::GpuBuffer;

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(32), 32);
        assert_eq!(next_power_of_two(33), 64);
        assert_eq!(next_power_of_two(512), 512);
        assert_eq!(next_power_of_two(513), 1024);
    }

    #[test]
    fn test_unified_workspace_allocation() {
        let mut ws = UnifiedLayerWorkspace::new();

        // Initially empty
        assert!(ws.norm1_out.is_none());
        assert_eq!(ws.allocation_count(), 0);

        // Allocate
        ws.ensure_capacity(32, 64, 128);
        assert!(ws.all_buffers_allocated());
        assert_eq!(ws.allocation_count(), 1);

        // Same size: reuse
        ws.ensure_capacity(32, 64, 128);
        assert_eq!(ws.allocation_count(), 1);

        // Different size: reallocate
        ws.ensure_capacity(64, 128, 256);
        assert_eq!(ws.allocation_count(), 2);

        // Much smaller: reallocate to save memory
        ws.ensure_capacity(16, 32, 128);
        assert_eq!(ws.allocation_count(), 3);
    }

    #[test]
    fn test_workspace_stats() {
        let mut ws = UnifiedLayerWorkspace::new();
        ws.ensure_capacity(32, 64, 128);

        let stats = ws.workspace_stats();
        assert_eq!(stats.buffer_count, 6); // 6 core buffers
        assert!(stats.expected_shape.is_some());
    }

    #[test]
    fn test_workspace_memory_estimation() {
        let ws = UnifiedLayerWorkspace::new();

        let estimated =
            estimate_allocation(32, 64, 128, false, false, false, false, false, false, false);
        let measured = ws.estimate_memory_usage(); // Should be 0 before allocation

        assert_eq!(measured, 0);
        assert!(estimated > 0);
    }

    #[test]
    fn test_exact_execution_capacity_uses_exact_shape() {
        let mut ws = UnifiedLayerWorkspace::new();
        ws.ensure_exact_execution_capacity(3, 5, 7);

        let norm1 = ws.norm1_out().expect("norm1_out should be allocated");
        assert_eq!(norm1.dim(), (3, 5));

        ws.set_streaming_state_enabled(true);
        ws.ensure_exact_execution_capacity(3, 5, 7);
        let state = ws
            .streaming_state()
            .expect("streaming_state should be allocated");
        assert_eq!(state.dim(), (3, 7));
    }

    #[test]
    fn test_streaming_state_reset() {
        let mut ws = UnifiedLayerWorkspace::new();
        ws.streaming_state = Some(Array2::ones((32, 128)));

        // Values should be 1.0
        assert!(ws.streaming_state().unwrap().iter().all(|&x| x == 1.0));

        ws.reset_streaming();

        // Values should now be 0.0
        assert!(ws.streaming_state().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_clear_workspace() {
        let mut ws = UnifiedLayerWorkspace::new();
        ws.ensure_capacity(32, 64, 128);
        assert!(ws.all_buffers_allocated());

        ws.clear_workspace();
        assert!(!ws.all_buffers_allocated());
        assert_eq!(ws.estimate_memory_usage(), 0);
    }

    #[test]
    fn test_diffusion_buffers_allocation() {
        let mut ws = UnifiedLayerWorkspace::new();

        // Initialize Diffusion-specific buffers
        ws.input_buffer = Some(Array2::zeros((1, 1)));
        ws.time_embed = Some(Array1::zeros(1));
        ws.film_modulation_scale = Some(Array2::zeros((1, 1)));
        ws.film_modulation_shift = Some(Array2::zeros((1, 1)));

        // Allocate
        ws.ensure_capacity(32, 64, 128);

        // Check that Diffusion buffers are allocated with correct shapes
        assert!(ws.input_buffer().is_some());
        assert!(ws.time_embed().is_some());
        assert!(ws.film_modulation_scale().is_some());
        assert!(ws.film_modulation_shift().is_some());

        // Verify shapes
        let input_buf = ws.input_buffer().unwrap();
        assert_eq!(input_buf.dim(), (32, 64)); // batch x seq with power-of-2 padding

        let film_scale = ws.film_modulation_scale().unwrap();
        assert_eq!(film_scale.dim(), (32, 4 * 128)); // batch x (4 * embed_dim) for gamma/beta pairs
    }

    #[test]
    fn test_diffusion_memory_estimation() {
        let mut ws = UnifiedLayerWorkspace::new();
        ws.ensure_capacity(32, 64, 128);

        // Add Diffusion buffers
        ws.input_buffer = Some(Array2::zeros((32, 64)));
        ws.time_embed = Some(Array1::zeros(128));
        ws.film_modulation_scale = Some(Array2::zeros((32, 4 * 128)));
        ws.film_modulation_shift = Some(Array2::zeros((32, 4 * 128)));
        ws.output_buffer = Some(Array2::zeros((32, 64)));

        let estimated = estimate_allocation(32, 64, 128, true, true, true, true, true, true, true);
        let measured = ws.estimate_memory_usage();

        assert!(measured > 0);
        assert!(estimated > 0);
        // Measured should account for all core buffers + Diffusion-specific
    }

    #[test]
    fn test_output_buffer_allocation() {
        let mut ws = UnifiedLayerWorkspace::new();

        // Initialize output buffer
        ws.output_buffer = Some(Array2::zeros((1, 1)));

        // Allocate
        ws.ensure_capacity(32, 64, 128);

        // Verify output buffer is allocated
        assert!(ws.output_buffer().is_some());
        let output_buf = ws.output_buffer().unwrap();
        assert_eq!(output_buf.dim(), (32, 64)); // batch x seq
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_gpu_context_buffer_set_get_clear() {
        let mut ws = UnifiedLayerWorkspace::new();
        let buffer = GpuBuffer {
            id: 123,
            size_bytes: 4096,
        };

        ws.set_gpu_context_buffer(Some(buffer));
        assert_eq!(ws.gpu_context_buffer(), Some(buffer));

        ws.clear_gpu_buffers();
        assert!(ws.gpu_context_buffer().is_none());
    }
}
