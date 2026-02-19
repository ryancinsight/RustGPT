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

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use ndarray::Array2;

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::common::errors::{ModelError, Result};

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::GpuDevice;

// ============================================================================
// Selective Scan Parameters
// ============================================================================

/// Parameters for selective scan kernels
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
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

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
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
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
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

    // Phase 5.6.5: GPU selective scan forward
    // For now, use CPU implementation (bridge pattern)
    // Will replace with GPU kernel in Phase 5.6.5+

    use ndarray::{Array1, linalg::general_mat_mul};

    let mut output = Array2::zeros((seq_len, embed_dim));
    // h_init is [batch, state_dim], we use the first batch element for state
    let mut h: Array1<f32> = h_init.row(0).to_owned();

    // Sequential scan (can be parallelized on GPU)
    for t in 0..seq_len {
        let x_t = input.row(t).to_owned();

        // h_t = A @ h_{t-1} + B @ x_t
        let bx = b.dot(&x_t);
        h = a.dot(&h);
        for (i, bx_i) in bx.iter().enumerate() {
            h[i] += bx_i;
        }

        // y_t = C @ h_t + D @ x_t
        let ch = c.dot(&h);
        let dx = d.dot(&x_t);
        for (i, (ch_i, dx_i)) in ch.iter().zip(dx.iter()).enumerate() {
            output[[t, i]] = ch_i + dx_i;
        }
    }

    // Return h as a 2D array for consistency with API
    let h_out = h.insert_axis(ndarray::Axis(0));
    Ok((output, h_out))
}

/// GPU-accelerated selective scan backward pass
///
/// Computes gradients for A, B, C, D matrices and input
///
/// # Returns
/// - `input_grads`: [seq_len, embed_dim]
/// - `a_grads`: [state_dim, state_dim]
/// - `b_grads`: [state_dim, embed_dim]
/// - `c_grads`: [embed_dim, state_dim]
/// - `d_grads`: [embed_dim, embed_dim]
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn selective_scan_backward_gpu(
    device: &mut GpuDevice,
    input: &Array2<f32>,        // [seq_len, embed_dim]
    output_grads: &Array2<f32>, // [seq_len, embed_dim]
    a: &Array2<f32>,            // [state_dim, state_dim]
    b: &Array2<f32>,            // [state_dim, embed_dim]
    c: &Array2<f32>,            // [embed_dim, state_dim]
    d: &Array2<f32>,            // [embed_dim, embed_dim]
    h_final: &Array2<f32>,      // [batch, state_dim]
    params: &SelectiveScanParams,
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

    // Phase 5.6.5: GPU selective scan backward
    // Placeholder for full implementation
    // TODO: Implement backward scan with gradient propagation

    let input_grads = Array2::zeros((seq_len, embed_dim));
    let a_grads = Array2::zeros((state_dim, state_dim));
    let b_grads = Array2::zeros((state_dim, embed_dim));
    let c_grads = Array2::zeros((embed_dim, state_dim));
    let d_grads = Array2::zeros((embed_dim, embed_dim));

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
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
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

    // Validate
    if w_f.dim() != (embed_dim, embed_dim)
        || w_r.dim() != (embed_dim, embed_dim)
        || w_o.dim() != (embed_dim, embed_dim)
    {
        return Err(ModelError::InvalidInput {
            message: format!("All weight matrices must be [{}, {}]", embed_dim, embed_dim),
        });
    }

    // Phase 5.6.5: GPU RG-LRU forward
    // For now, use CPU implementation (bridge pattern)

    let mut output = Array2::zeros((seq_len, embed_dim));
    let mut h = h_init.clone();

    // Sequential processing (can be parallelized on GPU)
    for t in 0..seq_len {
        let x_t = input.row(t).to_owned();

        // Forget gate
        let f_logits = w_f.dot(&x_t);
        let f_t = f_logits.mapv(|x| 1.0 / (1.0 + (-x).exp())); // sigmoid

        // New recurrent value
        let r_t = w_r.dot(&x_t);

        // Update hidden state
        h = h * &f_t + &r_t * (1.0 - &f_t);

        // Output gate
        let o_logits = w_o.dot(&x_t);
        let o_t = o_logits.mapv(|x| 1.0 / (1.0 + (-x).exp())); // sigmoid

        // Output
        let y_t = &h * &o_t;
        for (i, y) in y_t.iter().enumerate() {
            output[[t, i]] = *y;
        }
    }

    Ok((output, h))
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
