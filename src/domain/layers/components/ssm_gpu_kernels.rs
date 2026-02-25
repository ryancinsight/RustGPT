//! SSM (State Space Model) GPU Kernels (Phase 5.6.5)
//!
//! GPU-accelerated kernels for SSM-based architectures:
//! - Mamba: Selective scan forward/backward
//! - RG-LRU: Gated recurrent computation
//! - Mamba2: Gating-optimized selective scan
//!
//! ## Architecture
//!
//! Selective Scan: Core recurrent operation
//! ```
//! for t in 0..T:
//!     h_t = A @ h_{t-1} + B @ x_t
//!     y_t = C @ h_t + D @ x_t
//! ```
//!
//! ## Performance Targets (Phase 5.6.5)
//!
//! - Selective scan forward: 20x speedup (40ms → 2ms on 512 batch)
//! - Selective scan backward: 15x speedup (50ms → 3ms on 512 batch)
//! - RG-LRU forward: 15x speedup (30ms → 2ms on 512 batch)

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use ndarray::{Array1, Array2};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::common::errors::{ModelError, Result};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::GpuDevice;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute_backend::ComputeBackend;

// ============================================================================
// Selective Scan Parameters
// ============================================================================

/// Parameters for selective scan kernels
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[derive(Debug, Clone)]
pub struct SelectiveScanParams {
    /// Sequence length
    pub seq_len: usize,
    /// State dimension (hidden size)
    pub state_dim: usize,
    /// Embedding/input dimension
    pub embed_dim: usize,
    /// Batch size
    pub batch_size: usize,
    /// Number of SSM blocks (for Mamba2)
    pub num_blocks: usize,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl SelectiveScanParams {
    pub fn new(seq_len: usize, state_dim: usize, embed_dim: usize, batch_size: usize) -> Self {
        Self {
            seq_len,
            state_dim,
            embed_dim,
            batch_size,
            num_blocks: 1,
        }
    }

    pub fn with_blocks(mut self, num_blocks: usize) -> Self {
        self.num_blocks = num_blocks;
        self
    }
}

// ============================================================================
// Internal Helpers
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn rg_lru_project_gpu(
    device: &mut GpuDevice,
    input: &Array2<f32>,
    w_f: &Array2<f32>,
    w_r: &Array2<f32>,
    w_o: &Array2<f32>,
) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    let (seq_len, embed_dim) = input.dim();

    let input_slice = input.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru projection input must be contiguous".to_string(),
    })?;
    let wf_slice = w_f.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru projection W_f must be contiguous".to_string(),
    })?;
    let wr_slice = w_r.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru projection W_r must be contiguous".to_string(),
    })?;
    let wo_slice = w_o.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru projection W_o must be contiguous".to_string(),
    })?;

    let mut input_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut wf_buf = device.allocate_f32(embed_dim * embed_dim)?;
    let mut wr_buf = device.allocate_f32(embed_dim * embed_dim)?;
    let mut wo_buf = device.allocate_f32(embed_dim * embed_dim)?;
    let mut f_logits_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut r_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut o_logits_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut f_sig_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut o_sig_buf = device.allocate_f32(seq_len * embed_dim)?;

    device.upload(input_slice, &mut input_buf)?;
    device.upload(wf_slice, &mut wf_buf)?;
    device.upload(wr_slice, &mut wr_buf)?;
    device.upload(wo_slice, &mut wo_buf)?;

    device.gemm_f32(
        1.0,
        &input_buf,
        &wf_buf,
        0.0,
        &mut f_logits_buf,
        seq_len,
        embed_dim,
        embed_dim,
        false,
        true,
    )?;
    device.gemm_f32(
        1.0, &input_buf, &wr_buf, 0.0, &mut r_buf, seq_len, embed_dim, embed_dim, false, true,
    )?;
    device.gemm_f32(
        1.0,
        &input_buf,
        &wo_buf,
        0.0,
        &mut o_logits_buf,
        seq_len,
        embed_dim,
        embed_dim,
        false,
        true,
    )?;

    device.sigmoid(&f_logits_buf, &mut f_sig_buf, seq_len * embed_dim)?;
    device.sigmoid(&o_logits_buf, &mut o_sig_buf, seq_len * embed_dim)?;

    let mut f_sig_host = vec![0.0f32; seq_len * embed_dim];
    let mut r_host = vec![0.0f32; seq_len * embed_dim];
    let mut o_sig_host = vec![0.0f32; seq_len * embed_dim];
    device.download(&f_sig_buf, &mut f_sig_host)?;
    device.download(&r_buf, &mut r_host)?;
    device.download(&o_sig_buf, &mut o_sig_host)?;

    device.deallocate(input_buf);
    device.deallocate(wf_buf);
    device.deallocate(wr_buf);
    device.deallocate(wo_buf);
    device.deallocate(f_logits_buf);
    device.deallocate(r_buf);
    device.deallocate(o_logits_buf);
    device.deallocate(f_sig_buf);
    device.deallocate(o_sig_buf);

    let f_sig = Array2::from_shape_vec((seq_len, embed_dim), f_sig_host).map_err(|err| {
        ModelError::InvalidInput {
            message: format!("Failed to reshape F gate matrix: {err}"),
        }
    })?;
    let r_proj = Array2::from_shape_vec((seq_len, embed_dim), r_host).map_err(|err| {
        ModelError::InvalidInput {
            message: format!("Failed to reshape R projection matrix: {err}"),
        }
    })?;
    let o_sig = Array2::from_shape_vec((seq_len, embed_dim), o_sig_host).map_err(|err| {
        ModelError::InvalidInput {
            message: format!("Failed to reshape O gate matrix: {err}"),
        }
    })?;

    Ok((f_sig, r_proj, o_sig))
}

// ============================================================================
// Selective Scan Kernels
// ============================================================================

/// GPU-accelerated selective scan forward pass
///
/// Core recurrence:
/// ```
/// h_t = A @ h_{t-1} + B @ x_t
/// y_t = C @ h_t + D @ x_t
/// ```
///
/// # Parameters
/// - `input`: [seq_len, embed_dim]
/// - `A`: [state_dim, state_dim] - state transition matrix
/// - `B`: [state_dim, embed_dim] - input projection
/// - `C`: [embed_dim, state_dim] - output projection
/// - `D`: [embed_dim, embed_dim] - feedthrough matrix
/// - `h_init`: [batch, state_dim] - initial hidden state
///
/// # Returns
/// - `output`: [seq_len, embed_dim]
/// - `h_final`: [batch, state_dim] - final hidden state
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn selective_scan_forward_gpu(
    device: &mut GpuDevice,
    input: &Array2<f32>,  // [seq_len, embed_dim]
    a: &Array2<f32>,      // [state_dim, state_dim]
    b: &Array2<f32>,      // [state_dim, embed_dim]
    c: &Array2<f32>,      // [embed_dim, state_dim]
    d: &Array2<f32>,      // [embed_dim, embed_dim]
    h_init: &Array2<f32>, // [batch, state_dim]
    params: &SelectiveScanParams,
) -> Result<(Array2<f32>, Array2<f32>)> {
    // Validate dimensions
    let (seq_len, embed_dim) = input.dim();
    let (state_dim, _) = a.dim();

    if seq_len != params.seq_len || embed_dim != params.embed_dim {
        return Err(ModelError::DimensionMismatchDetailed {
            expected: format!("input: [{}, {}]", params.seq_len, params.embed_dim),
            got: format!("[{}, {}]", seq_len, embed_dim),
        });
    }

    // Phase 5.6.5: Validate matrices
    if a.dim() != (state_dim, state_dim) {
        return Err(ModelError::InvalidInput {
            message: "A matrix must be square [state_dim, state_dim]".to_string(),
        });
    }

    if b.dim() != (state_dim, embed_dim)
        || c.dim() != (embed_dim, state_dim)
        || d.dim() != (embed_dim, embed_dim)
    {
        return Err(ModelError::InvalidInput {
            message: "B, C, D matrices have inconsistent dimensions".to_string(),
        });
    }
    let input_slice = input.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_forward_gpu input must be contiguous".to_string(),
    })?;

    let mut output = Array2::zeros((seq_len, embed_dim));
    let a_slice = a.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_forward_gpu A must be contiguous".to_string(),
    })?;
    let b_slice = b.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_forward_gpu B must be contiguous".to_string(),
    })?;
    let c_slice = c.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_forward_gpu C must be contiguous".to_string(),
    })?;
    let d_slice = d.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_forward_gpu D must be contiguous".to_string(),
    })?;

    let h0 = h_init.row(0).to_owned();
    let h0_slice = h0.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_forward_gpu h_init row must be contiguous".to_string(),
    })?;

    if matches!(
        device.backend(),
        ComputeBackend::Vulkan | ComputeBackend::Npu
    ) {
        let mut input_buf = device.allocate_f32(seq_len * embed_dim)?;
        let mut a_buf = device.allocate_f32(state_dim * state_dim)?;
        let mut b_buf = device.allocate_f32(state_dim * embed_dim)?;
        let mut c_buf = device.allocate_f32(embed_dim * state_dim)?;
        let mut d_buf = device.allocate_f32(embed_dim * embed_dim)?;
        let mut h_init_buf = device.allocate_f32(state_dim)?;
        let mut output_buf = device.allocate_f32(seq_len * embed_dim)?;
        let mut h_final_buf = device.allocate_f32(state_dim)?;

        device.upload(input_slice, &mut input_buf)?;
        device.upload(a_slice, &mut a_buf)?;
        device.upload(b_slice, &mut b_buf)?;
        device.upload(c_slice, &mut c_buf)?;
        device.upload(d_slice, &mut d_buf)?;
        device.upload(h0_slice, &mut h_init_buf)?;

        device.selective_scan_forward(
            &input_buf,
            &a_buf,
            &b_buf,
            &c_buf,
            &d_buf,
            &h_init_buf,
            &mut output_buf,
            &mut h_final_buf,
            seq_len,
            state_dim,
            embed_dim,
        )?;

        let mut output_host = vec![0.0f32; seq_len * embed_dim];
        device.download(&output_buf, &mut output_host)?;
        output = Array2::from_shape_vec((seq_len, embed_dim), output_host).map_err(|err| {
            ModelError::InvalidInput {
                message: format!("Failed to reshape output in selective_scan_forward_gpu: {err}"),
            }
        })?;

        let mut h_host = vec![0.0f32; state_dim];
        device.download(&h_final_buf, &mut h_host)?;

        device.deallocate(input_buf);
        device.deallocate(a_buf);
        device.deallocate(b_buf);
        device.deallocate(c_buf);
        device.deallocate(d_buf);
        device.deallocate(h_init_buf);
        device.deallocate(output_buf);
        device.deallocate(h_final_buf);

        let h_out = Array2::from_shape_vec((1, state_dim), h_host).map_err(|err| {
            ModelError::InvalidInput {
                message: format!("Failed to reshape h_final in selective_scan_forward_gpu: {err}"),
            }
        })?;
        return Ok((output, h_out));
    }

    // Generic backend path: GPU matvec recurrence with per-step dispatch.
    let mut a_buf = device.allocate_f32(state_dim * state_dim)?;
    let mut b_buf = device.allocate_f32(state_dim * embed_dim)?;
    let mut c_buf = device.allocate_f32(embed_dim * state_dim)?;
    let mut d_buf = device.allocate_f32(embed_dim * embed_dim)?;
    let mut x_buf = device.allocate_f32(embed_dim)?;
    let mut ah_buf = device.allocate_f32(state_dim)?;
    let mut bx_buf = device.allocate_f32(state_dim)?;
    let mut h_prev_buf = device.allocate_f32(state_dim)?;
    let mut h_next_buf = device.allocate_f32(state_dim)?;
    let mut ch_buf = device.allocate_f32(embed_dim)?;
    let mut dx_buf = device.allocate_f32(embed_dim)?;
    let mut y_buf = device.allocate_f32(embed_dim)?;

    device.upload(a_slice, &mut a_buf)?;
    device.upload(b_slice, &mut b_buf)?;
    device.upload(c_slice, &mut c_buf)?;
    device.upload(d_slice, &mut d_buf)?;
    device.upload(h0_slice, &mut h_prev_buf)?;

    let mut y_host = vec![0.0f32; embed_dim];
    for t in 0..seq_len {
        let row_start = t * embed_dim;
        let row_end = row_start + embed_dim;
        device.upload(&input_slice[row_start..row_end], &mut x_buf)?;

        // ah = A @ h_prev
        device.gemm_f32(
            1.0,
            &a_buf,
            &h_prev_buf,
            0.0,
            &mut ah_buf,
            state_dim,
            1,
            state_dim,
            false,
            false,
        )?;
        // bx = B @ x_t
        device.gemm_f32(
            1.0,
            &b_buf,
            &x_buf,
            0.0,
            &mut bx_buf,
            state_dim,
            1,
            embed_dim,
            false,
            false,
        )?;
        // h_next = ah + bx
        device.axpy(1.0, &ah_buf, 1.0, &bx_buf, &mut h_next_buf, state_dim)?;

        // ch = C @ h_next
        device.gemm_f32(
            1.0,
            &c_buf,
            &h_next_buf,
            0.0,
            &mut ch_buf,
            embed_dim,
            1,
            state_dim,
            false,
            false,
        )?;
        // dx = D @ x_t
        device.gemm_f32(
            1.0,
            &d_buf,
            &x_buf,
            0.0,
            &mut dx_buf,
            embed_dim,
            1,
            embed_dim,
            false,
            false,
        )?;
        // y_t = ch + dx
        device.axpy(1.0, &ch_buf, 1.0, &dx_buf, &mut y_buf, embed_dim)?;

        device.download(&y_buf, &mut y_host)?;
        for j in 0..embed_dim {
            output[[t, j]] = y_host[j];
        }

        std::mem::swap(&mut h_prev_buf, &mut h_next_buf);
    }

    let mut h_host = vec![0.0f32; state_dim];
    device.download(&h_prev_buf, &mut h_host)?;

    device.deallocate(a_buf);
    device.deallocate(b_buf);
    device.deallocate(c_buf);
    device.deallocate(d_buf);
    device.deallocate(x_buf);
    device.deallocate(ah_buf);
    device.deallocate(bx_buf);
    device.deallocate(h_prev_buf);
    device.deallocate(h_next_buf);
    device.deallocate(ch_buf);
    device.deallocate(dx_buf);
    device.deallocate(y_buf);

    let h_out =
        Array2::from_shape_vec((1, state_dim), h_host).map_err(|err| ModelError::InvalidInput {
            message: format!("Failed to reshape h_final in selective_scan_forward_gpu: {err}"),
        })?;
    Ok((output, h_out))
}

/// GPU-accelerated selective scan backward pass
///
/// Computes gradients for A, B, C, D matrices and input using reverse-time traversal.
///
/// ## Algorithm
///
/// The backward pass traverses the sequence in reverse:
/// 1. Compute forward pass to cache hidden states
/// 2. Backpropagate gradients from output to hidden state
/// 3. Backpropagate through recurrence: dh_{t-1} = A^T @ dh_t
/// 4. Accumulate parameter gradients
///
/// # Returns
/// - `input_grads`: [seq_len, embed_dim]
/// - `a_grads`: [state_dim, state_dim]
/// - `b_grads`: [state_dim, embed_dim]
/// - `c_grads`: [embed_dim, state_dim]
/// - `d_grads`: [embed_dim, embed_dim]
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn selective_scan_backward_gpu(
    device: &mut GpuDevice,
    input: &Array2<f32>,        // [seq_len, embed_dim]
    output_grads: &Array2<f32>, // [seq_len, embed_dim]
    a: &Array2<f32>,            // [state_dim, state_dim]
    b: &Array2<f32>,            // [state_dim, embed_dim]
    c: &Array2<f32>,            // [embed_dim, state_dim]
    d: &Array2<f32>,            // [embed_dim, embed_dim]
    _h_final: &Array2<f32>,     // [batch, state_dim]
    _params: &SelectiveScanParams,
) -> Result<(
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
)> {
    let (seq_len, embed_dim) = input.dim();
    let (state_dim, _) = a.dim();

    // Validate dimensions
    if output_grads.dim() != (seq_len, embed_dim) {
        return Err(ModelError::DimensionMismatchDetailed {
            expected: format!("output_grads: [{}, {}]", seq_len, embed_dim),
            got: format!("{:?}", output_grads.dim()),
        });
    }
    if b.dim() != (state_dim, embed_dim)
        || c.dim() != (embed_dim, state_dim)
        || d.dim() != (embed_dim, embed_dim)
    {
        return Err(ModelError::InvalidInput {
            message: format!(
                "Expected B/C/D shapes [{state_dim},{embed_dim}], [{embed_dim},{state_dim}], [{embed_dim},{embed_dim}]"
            ),
        });
    }

    let input_slice = input.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_backward_gpu input must be contiguous".to_string(),
    })?;
    let output_grads_slice = output_grads
        .as_slice()
        .ok_or_else(|| ModelError::InvalidInput {
            message: "selective_scan_backward_gpu output_grads must be contiguous".to_string(),
        })?;

    // Step 1: Recompute hidden states for backward using GPU recurrence matvecs.
    let mut h_cache = Array2::<f32>::zeros((seq_len + 1, state_dim));
    if seq_len > 0 {
        let a_slice = a.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "selective_scan_backward_gpu A must be contiguous".to_string(),
        })?;
        let b_slice = b.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "selective_scan_backward_gpu B must be contiguous".to_string(),
        })?;

        let mut a_buf = device.allocate_f32(state_dim * state_dim)?;
        let mut b_buf = device.allocate_f32(state_dim * embed_dim)?;
        let mut x_buf = device.allocate_f32(embed_dim)?;
        let mut ah_buf = device.allocate_f32(state_dim)?;
        let mut bx_buf = device.allocate_f32(state_dim)?;
        let mut h_prev_buf = device.allocate_f32(state_dim)?;
        let mut h_next_buf = device.allocate_f32(state_dim)?;
        let mut h_hist_buf = if matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu)
        {
            Some(device.allocate_f32(seq_len * state_dim)?)
        } else {
            None
        };
        let mut input_seq_buf = if matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu)
        {
            let mut buf = device.allocate_f32(seq_len * embed_dim)?;
            device.upload(input_slice, &mut buf)?;
            Some(buf)
        } else {
            None
        };

        device.upload(a_slice, &mut a_buf)?;
        device.upload(b_slice, &mut b_buf)?;
        let h0 = vec![0.0f32; state_dim];
        device.upload(&h0, &mut h_prev_buf)?;

        let mut h_host = vec![0.0f32; state_dim];
        for t in 0..seq_len {
            let row_start = t * embed_dim;
            let row_end = row_start + embed_dim;
            if let Some(seq_buf) = input_seq_buf.as_ref() {
                device.copy_within_device_range(seq_buf, row_start, &mut x_buf, 0, embed_dim)?;
            } else {
                device.upload(&input_slice[row_start..row_end], &mut x_buf)?;
            }

            // ah = A @ h_prev
            device.gemm_f32(
                1.0,
                &a_buf,
                &h_prev_buf,
                0.0,
                &mut ah_buf,
                state_dim,
                1,
                state_dim,
                false,
                false,
            )?;
            // bx = B @ x_t
            device.gemm_f32(
                1.0,
                &b_buf,
                &x_buf,
                0.0,
                &mut bx_buf,
                state_dim,
                1,
                embed_dim,
                false,
                false,
            )?;
            // h_next = ah + bx
            device.axpy(1.0, &ah_buf, 1.0, &bx_buf, &mut h_next_buf, state_dim)?;

            if let Some(hist_buf) = h_hist_buf.as_mut() {
                device.copy_within_device_range(&h_next_buf, 0, hist_buf, t * state_dim, state_dim)?;
            } else {
                device.download(&h_next_buf, &mut h_host)?;
                for j in 0..state_dim {
                    h_cache[[t + 1, j]] = h_host[j];
                }
            }
            std::mem::swap(&mut h_prev_buf, &mut h_next_buf);
        }

        if let Some(hist_buf) = h_hist_buf.as_ref() {
            let mut hist_host = vec![0.0f32; seq_len * state_dim];
            device.download(hist_buf, &mut hist_host)?;
            for t in 0..seq_len {
                let row_start = t * state_dim;
                let row_end = row_start + state_dim;
                for (j, value) in hist_host[row_start..row_end].iter().enumerate() {
                    h_cache[[t + 1, j]] = *value;
                }
            }
        }

        device.deallocate(a_buf);
        device.deallocate(b_buf);
        device.deallocate(x_buf);
        device.deallocate(ah_buf);
        device.deallocate(bx_buf);
        device.deallocate(h_prev_buf);
        device.deallocate(h_next_buf);
        if let Some(buf) = input_seq_buf.take() {
            device.deallocate(buf);
        }
        if let Some(buf) = h_hist_buf.take() {
            device.deallocate(buf);
        }
    }

    // Step 2: Reverse recurrence for hidden-state gradients on GPU matvecs.
    let a_t = a.t().to_owned();
    let c_t = c.t().to_owned();
    let mut dh_seq = Array2::<f32>::zeros((seq_len, state_dim));
    if seq_len > 0 {
        let a_t_slice = a_t.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "selective_scan_backward_gpu A^T must be contiguous".to_string(),
        })?;
        let c_t_slice = c_t.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "selective_scan_backward_gpu C^T must be contiguous".to_string(),
        })?;

        let mut a_t_buf = device.allocate_f32(state_dim * state_dim)?;
        let mut c_t_buf = device.allocate_f32(state_dim * embed_dim)?;
        let mut dy_buf = device.allocate_f32(embed_dim)?;
        let mut ctdy_buf = device.allocate_f32(state_dim)?;
        let mut dh_acc_buf = device.allocate_f32(state_dim)?;
        let mut dh_prev_buf = device.allocate_f32(state_dim)?;
        let mut dh_next_buf = device.allocate_f32(state_dim)?;
        let mut dh_hist_buf =
            if matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu) {
                Some(device.allocate_f32(seq_len * state_dim)?)
            } else {
                None
            };
        let mut output_grads_seq_buf =
            if matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu) {
                let mut buf = device.allocate_f32(seq_len * embed_dim)?;
                device.upload(output_grads_slice, &mut buf)?;
                Some(buf)
            } else {
                None
            };
        let zero_state = vec![0.0f32; state_dim];

        device.upload(a_t_slice, &mut a_t_buf)?;
        device.upload(c_t_slice, &mut c_t_buf)?;
        device.upload(&zero_state, &mut dh_next_buf)?;

        let mut dh_host = vec![0.0f32; state_dim];
        for t in (0..seq_len).rev() {
            let row_start = t * embed_dim;
            let row_end = row_start + embed_dim;
            if let Some(seq_buf) = output_grads_seq_buf.as_ref() {
                device.copy_within_device_range(seq_buf, row_start, &mut dy_buf, 0, embed_dim)?;
            } else {
                device.upload(&output_grads_slice[row_start..row_end], &mut dy_buf)?;
            }

            // ctdy = C^T @ dy_t
            device.gemm_f32(
                1.0,
                &c_t_buf,
                &dy_buf,
                0.0,
                &mut ctdy_buf,
                state_dim,
                1,
                embed_dim,
                false,
                false,
            )?;
            // dh_acc = dh_next + ctdy
            device.axpy(
                1.0,
                &dh_next_buf,
                1.0,
                &ctdy_buf,
                &mut dh_acc_buf,
                state_dim,
            )?;

            if let Some(hist_buf) = dh_hist_buf.as_mut() {
                device.copy_within_device_range(
                    &dh_acc_buf,
                    0,
                    hist_buf,
                    t * state_dim,
                    state_dim,
                )?;
            } else {
                device.download(&dh_acc_buf, &mut dh_host)?;
                for j in 0..state_dim {
                    dh_seq[[t, j]] = dh_host[j];
                }
            }

            // dh_prev = A^T @ dh_acc
            device.gemm_f32(
                1.0,
                &a_t_buf,
                &dh_acc_buf,
                0.0,
                &mut dh_prev_buf,
                state_dim,
                1,
                state_dim,
                false,
                false,
            )?;
            std::mem::swap(&mut dh_next_buf, &mut dh_prev_buf);
        }

        if let Some(hist_buf) = dh_hist_buf.as_ref() {
            let mut hist_host = vec![0.0f32; seq_len * state_dim];
            device.download(hist_buf, &mut hist_host)?;
            for t in 0..seq_len {
                let row_start = t * state_dim;
                let row_end = row_start + state_dim;
                for (j, value) in hist_host[row_start..row_end].iter().enumerate() {
                    dh_seq[[t, j]] = *value;
                }
            }
        }

        device.deallocate(a_t_buf);
        device.deallocate(c_t_buf);
        device.deallocate(dy_buf);
        device.deallocate(ctdy_buf);
        device.deallocate(dh_acc_buf);
        device.deallocate(dh_prev_buf);
        device.deallocate(dh_next_buf);
        if let Some(buf) = output_grads_seq_buf.take() {
            device.deallocate(buf);
        }
        if let Some(buf) = dh_hist_buf.take() {
            device.deallocate(buf);
        }
    }

    // Step 3: Form cached state matrices [T, S] for batched gradient GEMMs.
    let h_t_mat = h_cache.slice(ndarray::s![1.., ..]).to_owned();
    let h_prev_mat = h_cache.slice(ndarray::s![0..seq_len, ..]).to_owned();

    // Step 4: GPU contractions for parameter and input gradients.
    // Reuse uploaded buffers across GEMMs to reduce transfer overhead.
    let h_t_slice = h_t_mat.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_backward_gpu h_t_mat must be contiguous".to_string(),
    })?;
    let h_prev_slice = h_prev_mat
        .as_slice()
        .ok_or_else(|| ModelError::InvalidInput {
            message: "selective_scan_backward_gpu h_prev_mat must be contiguous".to_string(),
        })?;
    let dh_seq_slice = dh_seq.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_backward_gpu dh_seq must be contiguous".to_string(),
    })?;
    let b_slice = b.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_backward_gpu B must be contiguous".to_string(),
    })?;
    let d_slice = d.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "selective_scan_backward_gpu D must be contiguous".to_string(),
    })?;

    let mut output_grads_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut input_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut h_t_buf = device.allocate_f32(seq_len * state_dim)?;
    let mut h_prev_buf = device.allocate_f32(seq_len * state_dim)?;
    let mut dh_seq_buf = device.allocate_f32(seq_len * state_dim)?;
    let mut b_buf = device.allocate_f32(state_dim * embed_dim)?;
    let mut d_buf = device.allocate_f32(embed_dim * embed_dim)?;
    device.upload(output_grads_slice, &mut output_grads_buf)?;
    device.upload(input_slice, &mut input_buf)?;
    device.upload(h_t_slice, &mut h_t_buf)?;
    device.upload(h_prev_slice, &mut h_prev_buf)?;
    device.upload(dh_seq_slice, &mut dh_seq_buf)?;
    device.upload(b_slice, &mut b_buf)?;
    device.upload(d_slice, &mut d_buf)?;

    let mut gemm_download = |lhs: &_,
                             rhs: &_,
                             m: usize,
                             n: usize,
                             k: usize,
                             trans_lhs: bool,
                             trans_rhs: bool|
     -> Result<Array2<f32>> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(Array2::zeros((m, n)));
        }
        let mut out_buf = device.allocate_f32(m * n)?;
        device.gemm_f32(
            1.0,
            lhs,
            rhs,
            0.0,
            &mut out_buf,
            m,
            n,
            k,
            trans_lhs,
            trans_rhs,
        )?;
        let mut host = vec![0.0f32; m * n];
        device.download(&out_buf, &mut host)?;
        device.deallocate(out_buf);
        Array2::from_shape_vec((m, n), host).map_err(|err| ModelError::InvalidInput {
            message: format!("selective_scan_backward_gpu GEMM reshape failed: {err}"),
        })
    };

    let c_grads = gemm_download(
        &output_grads_buf,
        &h_t_buf,
        embed_dim,
        state_dim,
        seq_len,
        true,
        false,
    )?;
    let d_grads = gemm_download(
        &output_grads_buf,
        &input_buf,
        embed_dim,
        embed_dim,
        seq_len,
        true,
        false,
    )?;
    let a_grads = gemm_download(
        &dh_seq_buf,
        &h_prev_buf,
        state_dim,
        state_dim,
        seq_len,
        true,
        false,
    )?;
    let b_grads = gemm_download(
        &dh_seq_buf,
        &input_buf,
        state_dim,
        embed_dim,
        seq_len,
        true,
        false,
    )?;
    let dx_from_b = gemm_download(
        &dh_seq_buf,
        &b_buf,
        seq_len,
        embed_dim,
        state_dim,
        false,
        false,
    )?;
    let dx_from_d = gemm_download(
        &output_grads_buf,
        &d_buf,
        seq_len,
        embed_dim,
        embed_dim,
        false,
        false,
    )?;

    device.deallocate(output_grads_buf);
    device.deallocate(input_buf);
    device.deallocate(h_t_buf);
    device.deallocate(h_prev_buf);
    device.deallocate(dh_seq_buf);
    device.deallocate(b_buf);
    device.deallocate(d_buf);
    let input_grads = &dx_from_b + &dx_from_d;

    Ok((input_grads, a_grads, b_grads, c_grads, d_grads))
}

// ============================================================================
// RG-LRU Specific Kernels
// ============================================================================

/// GPU-accelerated RG-LRU (Recurrent Gated Linear Recurrent Unit) forward pass
///
/// Combines gating and linear recurrent computation:
/// ```
/// f_t = sigmoid(W_f @ x_t + b_f)  // Forget gate
/// r_t = W_r @ x_t                 // New recurrent value
/// h_t = f_t * h_{t-1} + (1 - f_t) * r_t
/// y_t = h_t * sigmoid(W_o @ x_t)  // Output gating
/// ```
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn rg_lru_forward_gpu(
    device: &mut GpuDevice,
    input: &Array2<f32>,  // [seq_len, embed_dim]
    w_f: &Array2<f32>,    // [embed_dim, embed_dim]
    w_r: &Array2<f32>,    // [embed_dim, embed_dim]
    w_o: &Array2<f32>,    // [embed_dim, embed_dim]
    h_init: &Array2<f32>, // [batch, embed_dim]
    params: &SelectiveScanParams,
) -> Result<(Array2<f32>, Array2<f32>)> {
    let (seq_len, embed_dim) = input.dim();
    if seq_len != params.seq_len || embed_dim != params.embed_dim {
        return Err(ModelError::DimensionMismatchDetailed {
            expected: format!("input: [{}, {}]", params.seq_len, params.embed_dim),
            got: format!("[{}, {}]", seq_len, embed_dim),
        });
    }
    if h_init.ncols() != embed_dim || h_init.nrows() == 0 {
        return Err(ModelError::InvalidInput {
            message: format!(
                "h_init must have shape [batch>=1, {}], got {:?}",
                embed_dim,
                h_init.dim()
            ),
        });
    }

    // Validate
    if w_f.dim() != (embed_dim, embed_dim)
        || w_r.dim() != (embed_dim, embed_dim)
        || w_o.dim() != (embed_dim, embed_dim)
    {
        return Err(ModelError::InvalidInput {
            message: format!("All weight matrices must be [{}, {}]", embed_dim, embed_dim),
        });
    }

    let mut output = Array2::zeros((seq_len, embed_dim));
    let (f_sig, r_proj, o_sig) = rg_lru_project_gpu(device, input, w_f, w_r, w_o)?;
    let f_slice = f_sig.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_forward_gpu F gate matrix must be contiguous".to_string(),
    })?;
    let r_slice = r_proj.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_forward_gpu R projection matrix must be contiguous".to_string(),
    })?;
    let o_slice = o_sig.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_forward_gpu O gate matrix must be contiguous".to_string(),
    })?;
    let h0 = h_init.row(0).to_owned();
    let h0_slice = h0.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_forward_gpu h_init row must be contiguous".to_string(),
    })?;

    // Sequential recurrent update on GPU element-wise kernels:
    // h_t = h_{t-1} * f_t + r_t * (1 - f_t)
    // y_t = h_t * o_t
    let mut f_buf = device.allocate_f32(embed_dim)?;
    let mut r_buf = device.allocate_f32(embed_dim)?;
    let mut o_buf = device.allocate_f32(embed_dim)?;
    let mut h_prev_buf = device.allocate_f32(embed_dim)?;
    let mut h_next_buf = device.allocate_f32(embed_dim)?;
    let mut ones_buf = device.allocate_f32(embed_dim)?;
    let mut one_minus_f_buf = device.allocate_f32(embed_dim)?;
    let mut tmp1_buf = device.allocate_f32(embed_dim)?;
    let mut tmp2_buf = device.allocate_f32(embed_dim)?;
    let mut y_buf = device.allocate_f32(embed_dim)?;
    let mut y_hist_buf = if seq_len > 0
        && matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu)
    {
        Some(device.allocate_f32(seq_len * embed_dim)?)
    } else {
        None
    };
    let mut f_seq_buf = if seq_len > 0
        && matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu)
    {
        let mut buf = device.allocate_f32(seq_len * embed_dim)?;
        device.upload(f_slice, &mut buf)?;
        Some(buf)
    } else {
        None
    };
    let mut r_seq_buf = if seq_len > 0
        && matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu)
    {
        let mut buf = device.allocate_f32(seq_len * embed_dim)?;
        device.upload(r_slice, &mut buf)?;
        Some(buf)
    } else {
        None
    };
    let mut o_seq_buf = if seq_len > 0
        && matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu)
    {
        let mut buf = device.allocate_f32(seq_len * embed_dim)?;
        device.upload(o_slice, &mut buf)?;
        Some(buf)
    } else {
        None
    };
    let ones = vec![1.0f32; embed_dim];
    device.upload(h0_slice, &mut h_prev_buf)?;
    device.upload(&ones, &mut ones_buf)?;

    let mut y_host = vec![0.0f32; embed_dim];
    for t in 0..seq_len {
        let row_start = t * embed_dim;
        let row_end = row_start + embed_dim;
        if let Some(seq_buf) = f_seq_buf.as_ref() {
            device.copy_within_device_range(seq_buf, row_start, &mut f_buf, 0, embed_dim)?;
        } else {
            device.upload(&f_slice[row_start..row_end], &mut f_buf)?;
        }
        if let Some(seq_buf) = r_seq_buf.as_ref() {
            device.copy_within_device_range(seq_buf, row_start, &mut r_buf, 0, embed_dim)?;
        } else {
            device.upload(&r_slice[row_start..row_end], &mut r_buf)?;
        }
        if let Some(seq_buf) = o_seq_buf.as_ref() {
            device.copy_within_device_range(seq_buf, row_start, &mut o_buf, 0, embed_dim)?;
        } else {
            device.upload(&o_slice[row_start..row_end], &mut o_buf)?;
        }

        // tmp1 = h_prev * f
        device.mul(&h_prev_buf, &f_buf, &mut tmp1_buf, embed_dim)?;
        // one_minus_f = 1 - f
        device.axpy(
            -1.0,
            &f_buf,
            1.0,
            &ones_buf,
            &mut one_minus_f_buf,
            embed_dim,
        )?;
        // tmp2 = r * (1 - f)
        device.mul(&r_buf, &one_minus_f_buf, &mut tmp2_buf, embed_dim)?;
        // h_next = tmp1 + tmp2
        device.axpy(1.0, &tmp1_buf, 1.0, &tmp2_buf, &mut h_next_buf, embed_dim)?;
        // y = h_next * o
        device.mul(&h_next_buf, &o_buf, &mut y_buf, embed_dim)?;
        if let Some(hist_buf) = y_hist_buf.as_mut() {
            device.copy_within_device_range(&y_buf, 0, hist_buf, row_start, embed_dim)?;
        } else {
            device.download(&y_buf, &mut y_host)?;
            for i in 0..embed_dim {
                output[[t, i]] = y_host[i];
            }
        }

        std::mem::swap(&mut h_prev_buf, &mut h_next_buf);
    }

    if let Some(hist_buf) = y_hist_buf.as_ref() {
        let mut output_host = vec![0.0f32; seq_len * embed_dim];
        device.download(hist_buf, &mut output_host)?;
        output =
            Array2::from_shape_vec((seq_len, embed_dim), output_host).map_err(|err| {
                ModelError::InvalidInput {
                    message: format!("Failed to reshape RG-LRU output in forward pass: {err}"),
                }
            })?;
    }

    let mut h_host = vec![0.0f32; embed_dim];
    device.download(&h_prev_buf, &mut h_host)?;
    let h_out =
        Array2::from_shape_vec((1, embed_dim), h_host).map_err(|err| ModelError::InvalidInput {
            message: format!("Failed to reshape h_final in rg_lru_forward_gpu: {err}"),
        })?;

    device.deallocate(f_buf);
    device.deallocate(r_buf);
    device.deallocate(o_buf);
    device.deallocate(h_prev_buf);
    device.deallocate(h_next_buf);
    device.deallocate(ones_buf);
    device.deallocate(one_minus_f_buf);
    device.deallocate(tmp1_buf);
    device.deallocate(tmp2_buf);
    device.deallocate(y_buf);
    if let Some(buf) = f_seq_buf.take() {
        device.deallocate(buf);
    }
    if let Some(buf) = r_seq_buf.take() {
        device.deallocate(buf);
    }
    if let Some(buf) = o_seq_buf.take() {
        device.deallocate(buf);
    }
    if let Some(buf) = y_hist_buf.take() {
        device.deallocate(buf);
    }

    Ok((output, h_out))
}

/// GPU-accelerated RG-LRU backward pass.
///
/// Recomputes recurrent caches and runs gradient contractions through GPU GEMM.
///
/// # Returns
/// - `input_grads`: [seq_len, embed_dim]
/// - `wf_grads`: [embed_dim, embed_dim]
/// - `wr_grads`: [embed_dim, embed_dim]
/// - `wo_grads`: [embed_dim, embed_dim]
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn rg_lru_backward_gpu(
    device: &mut GpuDevice,
    input: &Array2<f32>,        // [seq_len, embed_dim]
    output_grads: &Array2<f32>, // [seq_len, embed_dim]
    w_f: &Array2<f32>,          // [embed_dim, embed_dim]
    w_r: &Array2<f32>,          // [embed_dim, embed_dim]
    w_o: &Array2<f32>,          // [embed_dim, embed_dim]
    h_init: &Array2<f32>,       // [batch, embed_dim]
    params: &SelectiveScanParams,
) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)> {
    let (seq_len, embed_dim) = input.dim();
    if seq_len != params.seq_len || embed_dim != params.embed_dim {
        return Err(ModelError::DimensionMismatchDetailed {
            expected: format!("input: [{}, {}]", params.seq_len, params.embed_dim),
            got: format!("[{}, {}]", seq_len, embed_dim),
        });
    }
    if output_grads.dim() != (seq_len, embed_dim) {
        return Err(ModelError::DimensionMismatchDetailed {
            expected: format!("output_grads: [{}, {}]", seq_len, embed_dim),
            got: format!("{:?}", output_grads.dim()),
        });
    }
    if h_init.ncols() != embed_dim || h_init.nrows() == 0 {
        return Err(ModelError::InvalidInput {
            message: format!(
                "h_init must have shape [batch>=1, {}], got {:?}",
                embed_dim,
                h_init.dim()
            ),
        });
    }
    if w_f.dim() != (embed_dim, embed_dim)
        || w_r.dim() != (embed_dim, embed_dim)
        || w_o.dim() != (embed_dim, embed_dim)
    {
        return Err(ModelError::InvalidInput {
            message: format!("All weight matrices must be [{}, {}]", embed_dim, embed_dim),
        });
    }

    // Shared projection path (GPU GEMM + gate activations).
    let (f_sig, r_proj, o_sig) = rg_lru_project_gpu(device, input, w_f, w_r, w_o)?;

    // Recompute recurrent hidden states on GPU element-wise kernels.
    let mut h_cache = Array2::<f32>::zeros((seq_len + 1, embed_dim));
    h_cache.row_mut(0).assign(&h_init.row(0));
    let f_slice = f_sig.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_backward_gpu F gate matrix must be contiguous".to_string(),
    })?;
    let r_slice = r_proj.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_backward_gpu R projection matrix must be contiguous".to_string(),
    })?;
    let h0 = h_init.row(0).to_owned();
    let h0_slice = h0.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_backward_gpu h_init row must be contiguous".to_string(),
    })?;

    let mut f_buf = device.allocate_f32(embed_dim)?;
    let mut r_buf = device.allocate_f32(embed_dim)?;
    let mut h_prev_buf = device.allocate_f32(embed_dim)?;
    let mut h_next_buf = device.allocate_f32(embed_dim)?;
    let mut ones_buf = device.allocate_f32(embed_dim)?;
    let mut one_minus_f_buf = device.allocate_f32(embed_dim)?;
    let mut tmp1_buf = device.allocate_f32(embed_dim)?;
    let mut tmp2_buf = device.allocate_f32(embed_dim)?;
    let mut h_hist_buf = if seq_len > 0
        && matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu)
    {
        Some(device.allocate_f32(seq_len * embed_dim)?)
    } else {
        None
    };
    let mut f_seq_buf = if seq_len > 0
        && matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu)
    {
        let mut buf = device.allocate_f32(seq_len * embed_dim)?;
        device.upload(f_slice, &mut buf)?;
        Some(buf)
    } else {
        None
    };
    let mut r_seq_buf = if seq_len > 0
        && matches!(device.backend(), ComputeBackend::Vulkan | ComputeBackend::Npu)
    {
        let mut buf = device.allocate_f32(seq_len * embed_dim)?;
        device.upload(r_slice, &mut buf)?;
        Some(buf)
    } else {
        None
    };
    let ones = vec![1.0f32; embed_dim];
    device.upload(h0_slice, &mut h_prev_buf)?;
    device.upload(&ones, &mut ones_buf)?;

    let mut h_host = vec![0.0f32; embed_dim];
    for t in 0..seq_len {
        let row_start = t * embed_dim;
        let row_end = row_start + embed_dim;
        if let Some(seq_buf) = f_seq_buf.as_ref() {
            device.copy_within_device_range(seq_buf, row_start, &mut f_buf, 0, embed_dim)?;
        } else {
            device.upload(&f_slice[row_start..row_end], &mut f_buf)?;
        }
        if let Some(seq_buf) = r_seq_buf.as_ref() {
            device.copy_within_device_range(seq_buf, row_start, &mut r_buf, 0, embed_dim)?;
        } else {
            device.upload(&r_slice[row_start..row_end], &mut r_buf)?;
        }

        // tmp1 = h_prev * f
        device.mul(&h_prev_buf, &f_buf, &mut tmp1_buf, embed_dim)?;
        // one_minus_f = 1 - f
        device.axpy(
            -1.0,
            &f_buf,
            1.0,
            &ones_buf,
            &mut one_minus_f_buf,
            embed_dim,
        )?;
        // tmp2 = r * (1 - f)
        device.mul(&r_buf, &one_minus_f_buf, &mut tmp2_buf, embed_dim)?;
        // h_next = tmp1 + tmp2
        device.axpy(1.0, &tmp1_buf, 1.0, &tmp2_buf, &mut h_next_buf, embed_dim)?;

        if let Some(hist_buf) = h_hist_buf.as_mut() {
            device.copy_within_device_range(&h_next_buf, 0, hist_buf, row_start, embed_dim)?;
        } else {
            device.download(&h_next_buf, &mut h_host)?;
            for j in 0..embed_dim {
                h_cache[[t + 1, j]] = h_host[j];
            }
        }
        std::mem::swap(&mut h_prev_buf, &mut h_next_buf);
    }

    if let Some(hist_buf) = h_hist_buf.as_ref() {
        let mut h_hist_host = vec![0.0f32; seq_len * embed_dim];
        device.download(hist_buf, &mut h_hist_host)?;
        for t in 0..seq_len {
            let row_start = t * embed_dim;
            let row_end = row_start + embed_dim;
            for (j, value) in h_hist_host[row_start..row_end].iter().enumerate() {
                h_cache[[t + 1, j]] = *value;
            }
        }
    }

    device.deallocate(f_buf);
    device.deallocate(r_buf);
    device.deallocate(h_prev_buf);
    device.deallocate(h_next_buf);
    device.deallocate(ones_buf);
    device.deallocate(one_minus_f_buf);
    device.deallocate(tmp1_buf);
    device.deallocate(tmp2_buf);
    if let Some(buf) = f_seq_buf.take() {
        device.deallocate(buf);
    }
    if let Some(buf) = r_seq_buf.take() {
        device.deallocate(buf);
    }
    if let Some(buf) = h_hist_buf.take() {
        device.deallocate(buf);
    }

    // Reverse-time recurrence gradients.
    let mut dh_next = Array1::<f32>::zeros(embed_dim);
    let mut df_logits = Array2::<f32>::zeros((seq_len, embed_dim));
    let mut dr = Array2::<f32>::zeros((seq_len, embed_dim));
    let mut do_logits = Array2::<f32>::zeros((seq_len, embed_dim));
    for t in (0..seq_len).rev() {
        let dy = output_grads.row(t);
        let h_t = h_cache.row(t + 1);
        let h_prev = h_cache.row(t);
        let f_t = f_sig.row(t);
        let r_t = r_proj.row(t);
        let o_t = o_sig.row(t);

        for j in 0..embed_dim {
            let dh = dy[j] * o_t[j] + dh_next[j];
            let do_t = dy[j] * h_t[j];
            do_logits[[t, j]] = do_t * o_t[j] * (1.0 - o_t[j]);

            let df_t = dh * (h_prev[j] - r_t[j]);
            df_logits[[t, j]] = df_t * f_t[j] * (1.0 - f_t[j]);
            dr[[t, j]] = dh * (1.0 - f_t[j]);
            dh_next[j] = dh * f_t[j];
        }
    }

    // Weight gradients: logits/r projections are X @ W^T, so dW = dY^T @ X.
    // Reuse uploaded buffers across GEMMs to reduce host<->device transfer overhead.
    let input_slice = input.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_backward_gpu input must be contiguous".to_string(),
    })?;
    let df_slice = df_logits
        .as_slice()
        .ok_or_else(|| ModelError::InvalidInput {
            message: "rg_lru_backward_gpu df_logits must be contiguous".to_string(),
        })?;
    let dr_slice = dr.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_backward_gpu dr must be contiguous".to_string(),
    })?;
    let do_slice = do_logits
        .as_slice()
        .ok_or_else(|| ModelError::InvalidInput {
            message: "rg_lru_backward_gpu do_logits must be contiguous".to_string(),
        })?;
    let wf_slice = w_f.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_backward_gpu w_f must be contiguous".to_string(),
    })?;
    let wr_slice = w_r.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_backward_gpu w_r must be contiguous".to_string(),
    })?;
    let wo_slice = w_o.as_slice().ok_or_else(|| ModelError::InvalidInput {
        message: "rg_lru_backward_gpu w_o must be contiguous".to_string(),
    })?;

    let mut input_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut df_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut dr_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut do_buf = device.allocate_f32(seq_len * embed_dim)?;
    let mut wf_buf = device.allocate_f32(embed_dim * embed_dim)?;
    let mut wr_buf = device.allocate_f32(embed_dim * embed_dim)?;
    let mut wo_buf = device.allocate_f32(embed_dim * embed_dim)?;
    device.upload(input_slice, &mut input_buf)?;
    device.upload(df_slice, &mut df_buf)?;
    device.upload(dr_slice, &mut dr_buf)?;
    device.upload(do_slice, &mut do_buf)?;
    device.upload(wf_slice, &mut wf_buf)?;
    device.upload(wr_slice, &mut wr_buf)?;
    device.upload(wo_slice, &mut wo_buf)?;

    let mut gemm_download = |lhs: &_,
                             rhs: &_,
                             m: usize,
                             n: usize,
                             k: usize,
                             trans_lhs: bool,
                             trans_rhs: bool|
     -> Result<Array2<f32>> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(Array2::zeros((m, n)));
        }
        let mut out_buf = device.allocate_f32(m * n)?;
        device.gemm_f32(
            1.0,
            lhs,
            rhs,
            0.0,
            &mut out_buf,
            m,
            n,
            k,
            trans_lhs,
            trans_rhs,
        )?;
        let mut host = vec![0.0f32; m * n];
        device.download(&out_buf, &mut host)?;
        device.deallocate(out_buf);
        Array2::from_shape_vec((m, n), host).map_err(|err| ModelError::InvalidInput {
            message: format!("rg_lru_backward_gpu GEMM reshape failed: {err}"),
        })
    };

    let wf_grads = gemm_download(
        &df_buf, &input_buf, embed_dim, embed_dim, seq_len, true, false,
    )?;
    let wr_grads = gemm_download(
        &dr_buf, &input_buf, embed_dim, embed_dim, seq_len, true, false,
    )?;
    let wo_grads = gemm_download(
        &do_buf, &input_buf, embed_dim, embed_dim, seq_len, true, false,
    )?;

    // Input gradients: dX = dF_logits@W_f + dR@W_r + dO_logits@W_o.
    let dx_f = gemm_download(
        &df_buf, &wf_buf, seq_len, embed_dim, embed_dim, false, false,
    )?;
    let dx_r = gemm_download(
        &dr_buf, &wr_buf, seq_len, embed_dim, embed_dim, false, false,
    )?;
    let dx_o = gemm_download(
        &do_buf, &wo_buf, seq_len, embed_dim, embed_dim, false, false,
    )?;

    device.deallocate(input_buf);
    device.deallocate(df_buf);
    device.deallocate(dr_buf);
    device.deallocate(do_buf);
    device.deallocate(wf_buf);
    device.deallocate(wr_buf);
    device.deallocate(wo_buf);
    let input_grads = &(&dx_f + &dx_r) + &dx_o;

    Ok((input_grads, wf_grads, wr_grads, wo_grads))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    use crate::domain::compute::GpuDevice;
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    use crate::domain::compute_backend::detect_available_and_compiled_gpu_backends;
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    use ndarray::Array2;

    #[test]
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_selective_scan_forward_shapes() {
        let seq_len = 64;
        let state_dim = 32;
        let embed_dim = 64;
        let batch_size = 4;

        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let a = Array2::<f32>::zeros((state_dim, state_dim));
        let b = Array2::<f32>::zeros((state_dim, embed_dim));
        let c = Array2::<f32>::zeros((embed_dim, state_dim));
        let d = Array2::<f32>::zeros((embed_dim, embed_dim));
        let h_init = Array2::<f32>::zeros((batch_size, state_dim));

        let params = SelectiveScanParams::new(seq_len, state_dim, embed_dim, batch_size);

        // Use auto-detection with no fallback (only backends that are compiled in)
        let backend = detect_available_and_compiled_gpu_backends()
            .into_iter()
            .next()
            .expect("No GPU backend available - this test requires a GPU");
        let mut device = GpuDevice::new(backend).unwrap();
        let result =
            selective_scan_forward_gpu(&mut device, &input, &a, &b, &c, &d, &h_init, &params);

        assert!(result.is_ok());
        let (output, h_final) = result.unwrap();

        assert_eq!(output.dim(), (seq_len, embed_dim));
        assert_eq!(h_final.dim(), (1, state_dim)); // h_final is now [1, state_dim]
    }

    #[test]
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_selective_scan_backward_shapes() {
        let seq_len = 64;
        let state_dim = 32;
        let embed_dim = 64;
        let batch_size = 4;

        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let output_grads = Array2::<f32>::zeros((seq_len, embed_dim));
        let a = Array2::<f32>::zeros((state_dim, state_dim));
        let b = Array2::<f32>::zeros((state_dim, embed_dim));
        let c = Array2::<f32>::zeros((embed_dim, state_dim));
        let d = Array2::<f32>::zeros((embed_dim, embed_dim));
        let h_final = Array2::<f32>::zeros((batch_size, state_dim));

        let params = SelectiveScanParams::new(seq_len, state_dim, embed_dim, batch_size);

        // Use auto-detection with no fallback (only backends that are compiled in)
        let backend = detect_available_and_compiled_gpu_backends()
            .into_iter()
            .next()
            .expect("No GPU backend available - this test requires a GPU");
        let mut device = GpuDevice::new(backend).unwrap();
        let result = selective_scan_backward_gpu(
            &mut device,
            &input,
            &output_grads,
            &a,
            &b,
            &c,
            &d,
            &h_final,
            &params,
        );

        assert!(result.is_ok());
        let (input_grads, a_grads, b_grads, c_grads, d_grads) = result.unwrap();

        assert_eq!(input_grads.dim(), (seq_len, embed_dim));
        assert_eq!(a_grads.dim(), (state_dim, state_dim));
        assert_eq!(b_grads.dim(), (state_dim, embed_dim));
        assert_eq!(c_grads.dim(), (embed_dim, state_dim));
        assert_eq!(d_grads.dim(), (embed_dim, embed_dim));
    }

    #[test]
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_rg_lru_forward_shapes() {
        let seq_len = 64;
        let embed_dim = 64;
        // Note: RG-LRU uses a single hidden state vector, not batched
        // The batch_size in params is for compatibility with SelectiveScanParams
        let batch_size = 1;

        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let w_f = Array2::<f32>::zeros((embed_dim, embed_dim));
        let w_r = Array2::<f32>::zeros((embed_dim, embed_dim));
        let w_o = Array2::<f32>::zeros((embed_dim, embed_dim));
        // Use single hidden state vector (1, embed_dim)
        let h_init = Array2::<f32>::zeros((1, embed_dim));

        let params = SelectiveScanParams::new(seq_len, embed_dim, embed_dim, batch_size);

        // Use auto-detection with no fallback (only backends that are compiled in)
        let backend = detect_available_and_compiled_gpu_backends()
            .into_iter()
            .next()
            .expect("No GPU backend available - this test requires a GPU");
        let mut device = GpuDevice::new(backend).unwrap();
        let result = rg_lru_forward_gpu(&mut device, &input, &w_f, &w_r, &w_o, &h_init, &params);

        assert!(result.is_ok());
        let (output, h_final) = result.unwrap();

        assert_eq!(output.dim(), (seq_len, embed_dim));
        // h_final should match h_init shape
        assert_eq!(h_final.dim(), (1, embed_dim));
    }

    #[test]
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_rg_lru_backward_shapes() {
        let seq_len = 64;
        let embed_dim = 64;
        let batch_size = 1;

        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let output_grads = Array2::<f32>::zeros((seq_len, embed_dim));
        let w_f = Array2::<f32>::zeros((embed_dim, embed_dim));
        let w_r = Array2::<f32>::zeros((embed_dim, embed_dim));
        let w_o = Array2::<f32>::zeros((embed_dim, embed_dim));
        let h_init = Array2::<f32>::zeros((1, embed_dim));

        let params = SelectiveScanParams::new(seq_len, embed_dim, embed_dim, batch_size);

        let backend = detect_available_and_compiled_gpu_backends()
            .into_iter()
            .next()
            .expect("No GPU backend available - this test requires a GPU");
        let mut device = GpuDevice::new(backend).unwrap();
        let result = rg_lru_backward_gpu(
            &mut device,
            &input,
            &output_grads,
            &w_f,
            &w_r,
            &w_o,
            &h_init,
            &params,
        );

        assert!(result.is_ok());
        let (input_grads, wf_grads, wr_grads, wo_grads) = result.unwrap();
        assert_eq!(input_grads.dim(), (seq_len, embed_dim));
        assert_eq!(wf_grads.dim(), (embed_dim, embed_dim));
        assert_eq!(wr_grads.dim(), (embed_dim, embed_dim));
        assert_eq!(wo_grads.dim(), (embed_dim, embed_dim));
    }

    #[test]
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_selective_scan_dimension_validation() {
        let seq_len = 64;
        let state_dim = 32;
        let embed_dim = 64;
        let batch_size = 4;

        let input = Array2::<f32>::zeros((seq_len, embed_dim + 1)); // Wrong!
        let a = Array2::<f32>::zeros((state_dim, state_dim));
        let b = Array2::<f32>::zeros((state_dim, embed_dim));
        let c = Array2::<f32>::zeros((embed_dim, state_dim));
        let d = Array2::<f32>::zeros((embed_dim, embed_dim));
        let h_init = Array2::<f32>::zeros((batch_size, state_dim));

        let params = SelectiveScanParams::new(seq_len, state_dim, embed_dim, batch_size);

        // Use auto-detection with no fallback (only backends that are compiled in)
        let backend = detect_available_and_compiled_gpu_backends()
            .into_iter()
            .next()
            .expect("No GPU backend available - this test requires a GPU");
        let mut device = GpuDevice::new(backend).unwrap();
        let result =
            selective_scan_forward_gpu(&mut device, &input, &a, &b, &c, &d, &h_init, &params);

        assert!(result.is_err(), "Should reject mismatched dimensions");
    }
}
