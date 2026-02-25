//! GPU Fused Forward Pass Kernel Composer
//!
//! Provides `FusedForwardComposer` — a helper that records a full dense layer
//! forward pass (linear → bias → layer-norm → activation) into a shared
//! `GpuDevice` command batch, then flushes once per layer instead of four
//! separate kernel submissions.
//!
//! ## Throughput Impact
//!
//! Without batching, each layer pays 4 round-trip CPU-GPU synchronization
//! costs (one per kernel launch). With `FusedForwardComposer` these are
//! collapsed to a single `flush()` call per layer, reducing GPU idle time
//! dramatically on small-to-medium sequence lengths where kernel launch
//! overhead dominates over compute time.
//!
//! ## Usage
//!
//! ```ignore
//! use crate::domain::compute::gpu_fused_forward::{FusedForwardComposer, LayerFusion};
//!
//! let mut fwd = FusedForwardComposer::new(device);
//! let output = fwd.dense_layer(
//!     &input_buf,
//!     &weight_buf,
//!     &bias_buf,
//!     &gamma_buf,
//!     &beta_buf,
//!     LayerFusion { activation: ActivationType::Gelu, has_layer_norm: true },
//!     batch, in_dim, out_dim,
//! )?;
//! // All 4 kernels submitted as one GPU batch.
//! ```

use crate::common::errors::Result;
use crate::domain::compute::{GpuBuffer, GpuDevice};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Activation function to apply as the final step of the fused forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    None,
    ReLU,
    GeLU,
    SiLU,
    Sigmoid,
}

/// Configuration for a fused layer forward pass.
#[derive(Debug, Clone, Copy)]
pub struct LayerFusion {
    /// Which activation to apply after bias add (or layer norm if enabled).
    pub activation: ActivationType,
    /// If true, apply layer normalization before the activation.
    pub has_layer_norm: bool,
    /// Layer norm epsilon (only used when `has_layer_norm = true`).
    pub layer_norm_eps: f32,
}

impl Default for LayerFusion {
    fn default() -> Self {
        Self {
            activation: ActivationType::GeLU,
            has_layer_norm: true,
            layer_norm_eps: 1e-5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Composer
// ─────────────────────────────────────────────────────────────────────────────

/// Fused forward pass composer.
///
/// Wraps a `GpuDevice` reference and issues all operations for a dense layer
/// (linear projection → bias add → optional layer norm → activation) in
/// deferred recording mode, submitting to the GPU in one batch.
pub struct FusedForwardComposer<'a> {
    device: &'a mut GpuDevice,
}

impl<'a> FusedForwardComposer<'a> {
    /// Create a new composer attached to the given GPU device.
    pub fn new(device: &'a mut GpuDevice) -> Self {
        Self { device }
    }

    /// Execute a fused dense-layer forward pass.
    ///
    /// Records:
    /// 1. `output = input @ weight^T`        (GEMM)
    /// 2. `output += bias`                   (broadcast add rows)
    /// 3. Optional `output = LayerNorm(output)` (if `fusion.has_layer_norm`)
    /// 4. Optional element-wise activation   (if `fusion.activation != None`)
    ///
    /// All 4 steps are batched into a single GPU submission via
    /// `begin_recording` / `flush`.
    ///
    /// # Returns
    /// A `GpuBuffer` of shape `[batch_size, out_dim]` containing the output.
    #[allow(clippy::too_many_arguments)]
    pub fn dense_layer(
        &mut self,
        input: &GpuBuffer,          // [batch_size, in_dim]
        weight: &GpuBuffer,         // [out_dim, in_dim]
        bias: &GpuBuffer,           // [out_dim]
        gamma: Option<&GpuBuffer>,  // [out_dim] (layer norm scale)
        beta_ln: Option<&GpuBuffer>,// [out_dim] (layer norm shift)
        fusion: LayerFusion,
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<GpuBuffer> {
        // Allocate output buffer
        let mut output = self.device.allocate_f32(batch_size * out_dim)?;

        // ── Begin batched recording ────────────────────────────────────────
        self.device.begin_recording();

        // 1. Linear: output = input @ weight^T
        self.device.gemm_f32(
            1.0,
            input,
            weight,
            0.0,
            &mut output,
            batch_size,
            out_dim,
            in_dim,
            false,
            true, // weight is [out_dim, in_dim] → transpose
        )?;

        // 2. Bias add
        self.device.broadcast_add_rows(&mut output, bias, batch_size, out_dim)?;

        // 3. Optional LayerNorm
        if fusion.has_layer_norm {
            if let (Some(g), Some(b)) = (gamma, beta_ln) {
                let mut normed = self.device.allocate_f32(batch_size * out_dim)?;
                self.device.layer_norm(
                    &output,
                    g, b,
                    &mut normed,
                    batch_size,
                    out_dim,
                    fusion.layer_norm_eps,
                )?;
                // Swap normed into output (deallocate old output)
                self.device.deallocate(output);
                output = normed;
            }
        }

        // 4. Activation
        match fusion.activation {
            ActivationType::None => {}
            ActivationType::ReLU => {
                let mut act = self.device.allocate_f32(batch_size * out_dim)?;
                self.device.relu(&output, &mut act, batch_size * out_dim)?;
                self.device.deallocate(output);
                output = act;
            }
            ActivationType::GeLU => {
                let mut act = self.device.allocate_f32(batch_size * out_dim)?;
                self.device.gelu(&output, &mut act, batch_size * out_dim)?;
                self.device.deallocate(output);
                output = act;
            }
            ActivationType::SiLU => {
                let mut act = self.device.allocate_f32(batch_size * out_dim)?;
                self.device.silu(&output, &mut act, batch_size * out_dim)?;
                self.device.deallocate(output);
                output = act;
            }
            ActivationType::Sigmoid => {
                let mut act = self.device.allocate_f32(batch_size * out_dim)?;
                self.device.sigmoid(&output, &mut act, batch_size * out_dim)?;
                self.device.deallocate(output);
                output = act;
            }
        }

        // ── Single GPU submission for all recorded ops ─────────────────────
        self.device.flush();

        Ok(output)
    }

    /// Fused attention score computation:
    /// `scores = (Q @ K^T) / sqrt(head_dim)` with optional causal masking.
    ///
    /// Records GEMM + scale + optional causal mask in a single batch.
    pub fn attention_scores(
        &mut self,
        q: &GpuBuffer,           // [batch * heads, seq, head_dim]
        k: &GpuBuffer,           // [batch * heads, seq, head_dim]
        scores: &mut GpuBuffer,  // [batch * heads, seq, seq]
        batch_heads: usize,
        seq_len: usize,
        head_dim: usize,
        scale: f32,              // typically 1/sqrt(head_dim)
        causal_mask: bool,
        num_heads: usize,
        batch_size: usize,
    ) -> Result<()> {
        self.device.begin_recording();

        // Q @ K^T scaled
        self.device.gemm_batched_f32(
            scale,
            q,
            k,
            0.0,
            scores,
            seq_len,
            seq_len,
            head_dim,
            batch_heads,
            [seq_len * head_dim, seq_len * head_dim, seq_len * seq_len],
            false,
            true,
        )?;

        // Optional causal mask
        if causal_mask {
            self.device.causal_mask_attention_scores(
                scores,
                batch_size,
                num_heads,
                seq_len,
                f32::NEG_INFINITY,
            )?;
        }

        self.device.flush();
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Async Data Prefetch
// ─────────────────────────────────────────────────────────────────────────────

/// Asynchronous data prefetcher.
///
/// Runs a background thread that tokenizes the next batch while the GPU
/// processes the current one, using a double-buffer (ping-pong) scheme.
///
/// This prevents GPU idle stalls caused by CPU-bound tokenization appearing
/// on the critical path.
pub struct DataPrefetcher {
    /// Pre-tokenized batches ready for upload.
    ///
    /// Capacity 2 = one batch in-flight on GPU, one being prepared on CPU.
    ready: std::sync::mpsc::Receiver<Vec<Vec<u32>>>,
    /// Sender side (kept alive to keep channel open).
    _worker_handle: Option<std::thread::JoinHandle<()>>,
}

impl DataPrefetcher {
    /// Create a prefetcher that tokenizes `texts` into batches of `batch_size`.
    ///
    /// Spawns a background thread that produces tokenized batches and sends
    /// them through a bounded channel (capacity 2 for double-buffering).
    pub fn new<F>(
        texts: Vec<String>,
        batch_size: usize,
        epochs: usize,
        tokenize: F,
    ) -> Self
    where
        F: Fn(&str) -> Vec<u32> + Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::sync_channel(2);

        let handle = std::thread::spawn(move || {
            for _epoch in 0..epochs {
                for chunk in texts.chunks(batch_size) {
                    let batch: Vec<Vec<u32>> = chunk
                        .iter()
                        .map(|s| tokenize(s.as_str()))
                        .collect();
                    if tx.send(batch).is_err() {
                        return; // Receiver dropped — training stopped.
                    }
                }
            }
        });

        Self {
            ready: rx,
            _worker_handle: Some(handle),
        }
    }

    /// Blocking: get the next pre-tokenized batch.
    ///
    /// Returns `None` when all epochs are exhausted.
    pub fn next_batch(&self) -> Option<Vec<Vec<u32>>> {
        self.ready.recv().ok()
    }
}
